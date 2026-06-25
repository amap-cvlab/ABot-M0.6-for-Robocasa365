from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from abot06.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .core import _CoreModel, _resolve_joint_steps
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .wan_video_dit import (
    WanVideoDiT,
    create_group_causal_attn_mask,
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)

logger = get_logger(__name__)


def extract_sample_action_horizon(
    sample: Dict[str, Any],
    *,
    batch_size: Optional[int] = None,
) -> Optional[int]:

    if "action_horizon" not in sample or sample["action_horizon"] is None:
        return None

    raw = sample["action_horizon"]
    values: list[int]
    if isinstance(raw, torch.Tensor):
        flat = raw.detach().reshape(-1)
        if flat.numel() == 0:
            raise ValueError("`sample['action_horizon']` must not be empty.")
        values = [int(v.item()) for v in flat]
    elif isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            raise ValueError("`sample['action_horizon']` must not be empty.")
        values = [int(v.item()) if isinstance(v, torch.Tensor) else int(v) for v in raw]
    else:
        values = [int(raw)]

    horizon = values[0]
    if horizon <= 0:
        raise ValueError(f"`sample['action_horizon']` must be positive, got {horizon}.")
    if any(v != horizon for v in values):
        raise ValueError(
            "`sample['action_horizon']` must be constant within a batch, "
            f"got values={values}"
        )
    if batch_size is not None and len(values) not in (1, int(batch_size)):
        raise ValueError(
            "`sample['action_horizon']` batch dimension mismatch: "
            f"got {len(values)} values for batch_size={batch_size}"
        )
    return horizon


def pad_for_3d_conv(x: torch.Tensor, kernel_size: Tuple[int, int, int]) -> torch.Tensor:


    if x.dim() != 5:
        raise ValueError(f"`pad_for_3d_conv` expects 5D `(B,C,T,H,W)`, got shape {tuple(x.shape)}")
    kt, kh, kw = int(kernel_size[0]), int(kernel_size[1]), int(kernel_size[2])
    if kt <= 0 or kh <= 0 or kw <= 0:
        raise ValueError(f"`kernel_size` entries must be positive, got {kernel_size}")
    _, _, t, h, w = x.shape
    pad_t = (-t) % kt
    pad_h = (-h) % kh
    pad_w = (-w) % kw
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def center_down_sample_3d(x: torch.Tensor, ratio: Tuple[int, int, int]) -> torch.Tensor:


    if x.dim() != 4:
        raise ValueError(f"`center_down_sample_3d` expects 4D `(T,H,W,D)`, got shape {tuple(x.shape)}")
    rt, rh, rw = int(ratio[0]), int(ratio[1]), int(ratio[2])
    if rt <= 0 or rh <= 0 or rw <= 0:
        raise ValueError(f"`ratio` entries must be positive, got {ratio}")
    if rt == 1 and rh == 1 and rw == 1:
        return x

    pooled = F.avg_pool3d(
        x.permute(3, 0, 1, 2).unsqueeze(0).contiguous(),
        kernel_size=(rt, rh, rw),
        stride=(rt, rh, rw),
    )
    return pooled.squeeze(0).permute(1, 2, 3, 0).contiguous()


def _downsample_freqs_cis(
    freqs_cis: torch.Tensor, ratio: Tuple[int, int, int]
) -> torch.Tensor:


    if not torch.is_complex(freqs_cis):
        raise ValueError("`_downsample_freqs_cis` expects complex-valued freqs.")
    real = center_down_sample_3d(freqs_cis.real.to(torch.float32), ratio)
    imag = center_down_sample_3d(freqs_cis.imag.to(torch.float32), ratio)
    return torch.complex(real, imag).to(freqs_cis.dtype)


class MemoryVideoDiT(WanVideoDiT):


    def __init__(
        self,
        *args: Any,
        memory: bool = False,
        memory_sizes: Tuple[int, int, int] = (16, 2, 1),
        memory_kernel_long: Tuple[int, int, int] = (4, 8, 8),
        memory_kernel_mid: Tuple[int, int, int] = (2, 4, 4),
        memory_kernel_current: Tuple[int, int, int] = (1, 2, 2),
        memory_pred_size: int = 2,
        memory_gain: bool = False,
        memory_zero_time: bool = True,
        **kwargs: Any,
    ) -> None:


        requested_fuse = kwargs.get("fuse_vae_embedding_in_latents", True)
        if memory and not requested_fuse:
            kwargs["fuse_vae_embedding_in_latents"] = True
        super().__init__(*args, **kwargs)

        self.memory = bool(memory)
        self.memory_sizes = tuple(int(s) for s in memory_sizes)
        self.memory_kernel_long = tuple(int(k) for k in memory_kernel_long)
        self.memory_kernel_mid = tuple(int(k) for k in memory_kernel_mid)
        self.memory_kernel_current = tuple(int(k) for k in memory_kernel_current)
        self.memory_pred_size = int(memory_pred_size)
        self.memory_gain = bool(memory_gain)
        self.memory_zero_time = bool(memory_zero_time)

        if not self.memory:
            return

        if len(self.memory_sizes) != 3:
            raise ValueError(
                f"`memory_sizes` must be a length-3 tuple `(long, mid, current)`, "
                f"got {self.memory_sizes}"
            )
        long_size, mid_size, curr_size = self.memory_sizes
        if long_size < 0 or mid_size < 0:
            raise ValueError(
                f"`memory_sizes` long/mid must be >= 0, got {self.memory_sizes}"
            )
        if curr_size < 1:
            raise ValueError(
                f"`memory_sizes` current (last element) must be >= 1, got {curr_size}"
            )
        if self.memory_pred_size <= 0:
            raise ValueError(f"`memory_pred_size` must be positive, got {self.memory_pred_size}")
        if not self.seperated_timestep:
            raise ValueError(
                "Memory requires `seperated_timestep=True` so each token can have its own timestep."
            )


        base_ps = tuple(self.patch_size)
        if self.memory_kernel_current != base_ps:
            raise ValueError(
                f"`memory_kernel_current` must equal `patch_size` {base_ps}, "
                f"got {self.memory_kernel_current}. Current segment shares `patch_embedding` with pred."
            )
        for name, kernel in [("long", self.memory_kernel_long), ("mid", self.memory_kernel_mid)]:
            if len(kernel) != 3 or any(k <= 0 for k in kernel):
                raise ValueError(f"`memory_patch_kernel_{name}` must be 3 positive ints, got {kernel}")
            kt, kh, kw = kernel
            if kh != kw:
                raise ValueError(
                    f"`memory_patch_kernel_{name}` spatial dims must be equal (square), got ({kh}, {kw})"
                )
            if kh % base_ps[1] != 0 or kw % base_ps[2] != 0 or kt % base_ps[0] != 0:
                raise ValueError(
                    f"`memory_patch_kernel_{name}` {kernel} must be integer multiples of "
                    f"`patch_size` {base_ps}"
                )


        base_t, base_h, _ = base_ps
        self._long_temporal_ratio = self.memory_kernel_long[0] // base_t
        self._long_spatial_ratio = self.memory_kernel_long[1] // base_h
        self._mid_temporal_ratio = self.memory_kernel_mid[0] // base_t
        self._mid_spatial_ratio = self.memory_kernel_mid[1] // base_h


        if memory and not requested_fuse:
            self.fuse_vae_embedding_in_latents = False


        self.patch_mid = nn.Conv3d(
            self.in_dim, self.hidden_dim,
            kernel_size=self.memory_kernel_mid, stride=self.memory_kernel_mid,
        )
        self.patch_long = nn.Conv3d(
            self.in_dim, self.hidden_dim,
            kernel_size=self.memory_kernel_long, stride=self.memory_kernel_long,
        )
        self._init_memory_patches_from_patch_embedding()

        if self.memory_gain:
            num_layers = len(self.blocks)

            self.history_key_scale_logit = nn.Parameter(
                torch.ones(num_layers, self.num_heads)
            )

    def _init_memory_patches_from_patch_embedding(self) -> None:


        weight = self.patch_embedding.weight.detach().clone()
        bias = self.patch_embedding.bias.detach().clone()
        base_ps = tuple(self.patch_size)
        if weight.shape[2:] != base_ps:
            raise ValueError(
                f"Expected `patch_embedding` kernel `{base_ps}`, got {tuple(weight.shape[2:])}"
            )
        rt_m, rs_m = self._mid_temporal_ratio, self._mid_spatial_ratio
        rt_l, rs_l = self._long_temporal_ratio, self._long_spatial_ratio
        mid_factor = float(rt_m * rs_m * rs_m)
        long_factor = float(rt_l * rs_l * rs_l)
        with torch.no_grad():
            self.patch_mid.weight.copy_(
                repeat(weight, f"d c t h w -> d c (t {rt_m}) (h {rs_m}) (w {rs_m})") / mid_factor
            )
            self.patch_mid.bias.copy_(bias)
            self.patch_long.weight.copy_(
                repeat(weight, f"d c t h w -> d c (t {rt_l}) (h {rs_l}) (w {rs_l})") / long_factor
            )
            self.patch_long.bias.copy_(bias)

    def build_memory_self_attn_mask(
        self,
        L_long: int,
        L_mid: int,
        L_curr: int,
        L_pred: int,
        tokens_per_frame_pred: int,
        device: torch.device,
    ) -> torch.Tensor:


        if min(L_long, L_mid, L_curr, L_pred) < 0:
            raise ValueError(
                f"All segment lengths must be non-negative, got long={L_long}, mid={L_mid}, "
                f"curr={L_curr}, pred={L_pred}"
            )
        L_hist = L_long + L_mid + L_curr
        total = L_hist + L_pred
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        if L_hist > 0:
            mask[:L_hist, :L_hist] = True
            if L_pred > 0:
                mask[L_hist:, :L_hist] = True
        if L_pred > 0:
            mask[L_hist:, L_hist:] = self.build_video_to_video_mask(
                video_seq_len=L_pred,
                video_tokens_per_frame=tokens_per_frame_pred,
                device=device,
            )
        return mask

    def _build_segment_freqs(
        self,
        frame_indices: torch.Tensor,
        h: int,
        w: int,
        spatial_ratio: int,
        temporal_ratio: int,
    ) -> torch.Tensor:


        f_freqs_cis, h_freqs_cis, w_freqs_cis = self.freqs
        device = frame_indices.device
        f_part = f_freqs_cis.to(device=device).index_select(0, frame_indices)
        h_part = h_freqs_cis.to(device=device)[:h]
        w_part = w_freqs_cis.to(device=device)[:w]
        t_dense = int(frame_indices.shape[0])
        f_grid = f_part.view(t_dense, 1, 1, -1).expand(t_dense, h, w, -1)
        h_grid = h_part.view(1, h, 1, -1).expand(t_dense, h, w, -1)
        w_grid = w_part.view(1, 1, w, -1).expand(t_dense, h, w, -1)
        dense = torch.cat([f_grid, h_grid, w_grid], dim=-1)
        if temporal_ratio > 1 or spatial_ratio > 1:

            dense_real = dense.real.to(torch.float32).permute(3, 0, 1, 2).unsqueeze(0)
            dense_imag = dense.imag.to(torch.float32).permute(3, 0, 1, 2).unsqueeze(0)
            pad_t = (-t_dense) % temporal_ratio
            pad_h = (-h) % spatial_ratio
            pad_w = (-w) % spatial_ratio
            if pad_t or pad_h or pad_w:
                dense_real = F.pad(dense_real, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
                dense_imag = F.pad(dense_imag, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
            kernel = (temporal_ratio, spatial_ratio, spatial_ratio)
            real_pool = F.avg_pool3d(dense_real, kernel_size=kernel, stride=kernel)
            imag_pool = F.avg_pool3d(dense_imag, kernel_size=kernel, stride=kernel)
            real_pool = real_pool.squeeze(0).permute(1, 2, 3, 0).contiguous()
            imag_pool = imag_pool.squeeze(0).permute(1, 2, 3, 0).contiguous()
            dense = torch.complex(real_pool, imag_pool).to(f_freqs_cis.dtype)
        seg_t, seg_h, seg_w, _ = dense.shape
        return dense.reshape(seg_t * seg_h * seg_w, 1, -1)

    def _validate_memory_inputs(
        self,
        x: torch.Tensor,
        history_long: torch.Tensor,
        history_mid: torch.Tensor,
        history_short: torch.Tensor,
    ) -> None:
        if x.dim() != 5 or history_long.dim() != 5 or history_mid.dim() != 5 or history_short.dim() != 5:
            raise ValueError(
                "All Memory tensors must be 5D `(B,C,T,H,W)`; got "
                f"x={tuple(x.shape)}, long={tuple(history_long.shape)}, "
                f"mid={tuple(history_mid.shape)}, short={tuple(history_short.shape)}"
            )
        if x.shape[2] != self.memory_pred_size:
            raise ValueError(
                f"`x` must have T={self.memory_pred_size} (pred), got T={x.shape[2]}"
            )
        long_size, mid_size, curr_size = self.memory_sizes
        if history_long.shape[2] != long_size:
            raise ValueError(f"`history_long` T must be {long_size}, got {history_long.shape[2]}")
        if history_mid.shape[2] != mid_size:
            raise ValueError(f"`history_mid` T must be {mid_size}, got {history_mid.shape[2]}")
        if history_short.shape[2] != curr_size:
            raise ValueError(
                f"`history_short` T must be {curr_size}, got {history_short.shape[2]}"
            )
        ref = (x.shape[0], x.shape[1], x.shape[3], x.shape[4])
        for name, tensor in (
            ("history_long", history_long),
            ("history_mid", history_mid),
            ("history_short", history_short),
        ):
            if (tensor.shape[0], tensor.shape[1], tensor.shape[3], tensor.shape[4]) != ref:
                raise ValueError(
                    f"`{name}` must share `(B, C, H, W)` with `x` `{ref}`, "
                    f"got {(tensor.shape[0], tensor.shape[1], tensor.shape[3], tensor.shape[4])}"
                )

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        history_long: Optional[torch.Tensor] = None,
        history_mid: Optional[torch.Tensor] = None,
        history_short: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        no_history = history_long is None and history_mid is None and history_short is None
        if (not self.memory) or no_history:
            return super().pre_dit(
                x=x,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                action=action,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                control_camera_latents_input=control_camera_latents_input,
            )

        if history_long is None or history_mid is None or history_short is None:
            raise ValueError(
                "Memory mode requires all three of `history_long`, `history_mid`, `history_short`."
            )
        if fuse_vae_embedding_in_latents:
            raise ValueError(
                "Memory mode is incompatible with `fuse_vae_embedding_in_latents=True`."
            )
        if control_camera_latents_input is not None:
            raise NotImplementedError("Memory mode does not support `control_camera_latents_input` yet.")


        x, timestep, context_mask = self._validate_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )
        self._validate_memory_inputs(x, history_long, history_mid, history_short)

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame_pred = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)


        x_pred = self.patchify(x)
        x_short = self.patch_embedding(history_short)

        f_pred, h, w = x_pred.shape[2], x_pred.shape[3], x_pred.shape[4]
        f_short, h_short, w_short = x_short.shape[2], x_short.shape[3], x_short.shape[4]
        if (h_short, w_short) != (h, w):
            raise ValueError(
                f"`patch_short` spatial shape ({h_short},{w_short}) must match pred ({h},{w})."
            )

        long_size, mid_size, curr_size = self.memory_sizes
        device = x.device
        segments_tokens: list[torch.Tensor] = []
        segments_freqs: list[torch.Tensor] = []


        if long_size > 0:
            x_long_in = pad_for_3d_conv(history_long, self.memory_kernel_long)
            x_long = self.patch_long(x_long_in)
            f_long, h_long, w_long = x_long.shape[2], x_long.shape[3], x_long.shape[4]
            L_long = f_long * h_long * w_long
            segments_tokens.append(rearrange(x_long, "b d t h w -> b (t h w) d"))
            idx_long = torch.arange(0, long_size, device=device)
            freqs_long = self._build_segment_freqs(
                frame_indices=idx_long, h=h, w=w,
                spatial_ratio=self._long_spatial_ratio,
                temporal_ratio=self._long_temporal_ratio,
            )
            if freqs_long.shape[0] != L_long:
                raise RuntimeError(
                    f"long freqs len {freqs_long.shape[0]} != L_long {L_long} (downsample mismatch)"
                )
            segments_freqs.append(freqs_long)
        else:
            L_long = 0


        if mid_size > 0:
            x_mid_in = pad_for_3d_conv(history_mid, self.memory_kernel_mid)
            x_mid = self.patch_mid(x_mid_in)
            f_mid, h_mid, w_mid = x_mid.shape[2], x_mid.shape[3], x_mid.shape[4]
            L_mid = f_mid * h_mid * w_mid
            segments_tokens.append(rearrange(x_mid, "b d t h w -> b (t h w) d"))
            idx_mid = torch.arange(long_size, long_size + mid_size, device=device)
            freqs_mid = self._build_segment_freqs(
                frame_indices=idx_mid, h=h, w=w,
                spatial_ratio=self._mid_spatial_ratio,
                temporal_ratio=self._mid_temporal_ratio,
            )
            if freqs_mid.shape[0] != L_mid:
                raise RuntimeError(
                    f"mid freqs len {freqs_mid.shape[0]} != L_mid {L_mid} (downsample mismatch)"
                )
            segments_freqs.append(freqs_mid)
        else:
            L_mid = 0


        L_curr = f_short * h_short * w_short
        segments_tokens.append(rearrange(x_short, "b d t h w -> b (t h w) d"))
        idx_curr = torch.arange(
            long_size + mid_size, long_size + mid_size + curr_size, device=device
        )
        freqs_short = self._build_segment_freqs(
            frame_indices=idx_curr, h=h, w=w, spatial_ratio=1, temporal_ratio=1
        )
        segments_freqs.append(freqs_short)


        L_pred = f_pred * h * w
        segments_tokens.append(rearrange(x_pred, "b d t h w -> b (t h w) d"))
        pred_start = long_size + mid_size + curr_size
        idx_pred = torch.arange(pred_start, pred_start + f_pred, device=device)
        freqs_pred = self._build_segment_freqs(
            frame_indices=idx_pred, h=h, w=w, spatial_ratio=1, temporal_ratio=1
        )
        segments_freqs.append(freqs_pred)

        L_hist = L_long + L_mid + L_curr
        total_seq = L_hist + L_pred
        tokens = torch.cat(segments_tokens, dim=1).contiguous()
        freqs = torch.cat(segments_freqs, dim=0).to(device)


        token_timesteps = torch.empty(
            (batch_size, total_seq), dtype=timestep.dtype, device=timestep.device
        )
        if self.memory_zero_time:
            if L_hist > 0:
                token_timesteps[:, :L_hist] = 0
        else:
            token_timesteps[:, :L_hist] = timestep.view(batch_size, 1)
        token_timesteps[:, L_hist:] = timestep.view(batch_size, 1)
        t_emb = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        ).reshape(batch_size, total_seq, self.hidden_dim)
        t_mod = self.time_projection(t_emb).unflatten(2, (6, self.hidden_dim))
        t_for_head = t_emb[:, L_hist:, :].contiguous()


        context = self.text_embedding(context)
        context_len = context.shape[1]
        if self.action_conditioned and action is not None:
            num_temporal_groups = self.memory_pred_size - 1
            if num_temporal_groups <= 0:
                raise ValueError(
                    "Action-conditioned Memory requires `memory_pred_size >= 2`, "
                    f"got {self.memory_pred_size}."
                )
            if action.shape[1] % num_temporal_groups != 0:
                raise ValueError(
                    f"`action` length {action.shape[1]} must be divisible by `pred-1`={num_temporal_groups}."
                )
            action_len = action.shape[1]
            action_emb = self.action_embedding(action)
            action_pos = sinusoidal_embedding_1d(
                self.hidden_dim,
                torch.arange(action_len, device=action_emb.device),
            )
            action_emb = action_emb + action_pos.unsqueeze(0)
            context = torch.cat([context, action_emb], dim=1)
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame_pred,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(device=context.device)
            final_context_mask = torch.zeros(
                (batch_size, total_seq, context.shape[1]), dtype=torch.bool, device=context.device
            )
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(
                -1, total_seq, -1
            )

            pred_action_start = L_hist + tokens_per_frame_pred
            final_context_mask[:, pred_action_start:total_seq, context_len:] = action_group_mask.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            final_context_mask = final_context_mask
        else:
            if self.action_conditioned and action is None:
                raise ValueError(
                    "Action-conditioned Memory requires non-null `action`. "
                    "Pure-text fallback (`f==1`) is not supported in Memory mode."
                )
            final_context_mask = context_mask.unsqueeze(1).expand(-1, total_seq, -1)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t_for_head,
            "t_mod": t_mod,
            "context": context,
            "context_mask": final_context_mask,
            "meta": {
                "grid_size": (f_pred, h, w),
                "tokens_per_frame": tokens_per_frame_pred,
                "batch_size": batch_size,
                "memory": {
                    "L_long": L_long,
                    "L_mid": L_mid,
                    "L_curr": L_curr,
                    "L_pred": L_pred,
                    "L_hist": L_hist,
                    "tokens_per_frame_pred": tokens_per_frame_pred,
                    "grid_size_pred": (f_pred, h, w),
                },
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        memory_meta = pre_state["meta"].get("memory")
        if memory_meta is None:
            return super().post_dit(x_tokens, pre_state)
        L_hist = int(memory_meta["L_hist"])
        f, h, w = memory_meta["grid_size_pred"]
        x_tokens = x_tokens[:, L_hist:, :].contiguous()
        x = self.head(x_tokens, pre_state["t"])
        return self.unpatchify(x, (f, h, w))


class MemoryMoT(MoT):


    def _build_expert_attention_io(
        self,
        expert: nn.Module,
        block: nn.Module,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
        layer_idx: int = 0,
        memory_seq_len: int = 0,
    ):
        q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_gc = (
            super()._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=freqs,
                t_mod=t_mod,
            )
        )
        if (
            memory_seq_len > 0
            and getattr(expert, "memory", False)
            and getattr(expert, "memory_gain", False)
            and hasattr(expert, "history_key_scale_logit")
        ):
            logit = expert.history_key_scale_logit[layer_idx]
            scale = (1.0 + 9.0 * torch.sigmoid(logit)).to(k.dtype).to(k.device)
            B, S, D = k.shape
            H = block.num_heads
            Dh = D // H
            k = k.view(B, S, H, Dh).contiguous()
            scaled_hist = k[:, :memory_seq_len] * scale.view(1, 1, H, 1)
            k = torch.cat(
                [scaled_hist, k[:, memory_seq_len:]], dim=1
            ).reshape(B, S, D)
        return (
            q,
            k,
            v,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gc,
        )

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        memory_seq_lens: Optional[Dict[str, int]] = None,
    ):
        if memory_seq_lens is None:
            memory_seq_lens = {}

        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks: list[torch.Tensor] = []
            k_chunks: list[torch.Tensor] = []
            v_chunks: list[torch.Tensor] = []
            cached: Dict[str, dict] = {}
            seq_lens: list[int] = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gc,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                    layer_idx=layer_idx,
                    memory_seq_len=int(memory_seq_lens.get(name, 0)),
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gc,
                }

            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(
                q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask
            )

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )
                tokens_all[name] = updated_tokens
                start = end

        return tokens_all

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        memory_seq_lens: Optional[Dict[str, int]] = None,
    ) -> list[dict[str, torch.Tensor]]:
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        video_hist_len = int((memory_seq_lens or {}).get("video", 0))
        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gc,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
                layer_idx=layer_idx,
                memory_seq_len=video_hist_len,
            )
            mixed = self._mixed_attention(
                q_cat=q, k_cat=k, v_cat=v, attention_mask=video_attention_mask
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gc,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        memory_seq_lens: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]
        action_hist_len = int((memory_seq_lens or {}).get("action", 0))

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gc,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
                layer_idx=layer_idx,
                memory_seq_len=action_hist_len,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )
            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gc,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x


class _Base(_CoreModel):


    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        memory: bool = False,
        memory_sizes: Tuple[int, int, int] = (16, 2, 1),
        memory_kernel_long: Tuple[int, int, int] = (4, 8, 8),
        memory_kernel_mid: Tuple[int, int, int] = (2, 4, 4),
        memory_kernel_current: Tuple[int, int, int] = (1, 2, 2),
        memory_pred_size: int = 2,
        memory_gain: bool = False,
        memory_zero_time: bool = True,
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
        )
        self.memory = bool(memory)
        self.memory_sizes = tuple(int(s) for s in memory_sizes)
        self.memory_pred_size = int(memory_pred_size)
        self.memory_gain = bool(memory_gain)
        self.memory_zero_time = bool(memory_zero_time)

        if self.memory:
            if not isinstance(self.video_expert, MemoryVideoDiT):
                raise ValueError(
                    "`memory=True` requires `video_expert` to be an MemoryVideoDiT instance."
                )
            if not isinstance(self.mot, MemoryMoT):
                raise ValueError(
                    "`memory=True` requires `mot` to be an MemoryMoT instance."
                )
            if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
                raise ValueError(
                    "Memory mode is incompatible with `video_expert.fuse_vae_embedding_in_latents=True`."
                )


    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: Optional[Dict[str, Any]] = None,
        action_dit_config: Optional[Dict[str, Any]] = None,
        action_dit_pretrained_path: Optional[str] = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        memory: bool = False,
        memory_sizes: Tuple[int, int, int] = (16, 2, 1),
        memory_kernel_long: Tuple[int, int, int] = (4, 8, 8),
        memory_kernel_mid: Tuple[int, int, int] = (2, 4, 4),
        memory_kernel_current: Tuple[int, int, int] = (1, 2, 2),
        memory_pred_size: int = 2,
        memory_gain: bool = False,
        memory_zero_time: bool = True,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for _Base.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for _Base.")


        loader_dit_config = dict(video_dit_config)
        if memory:
            user_fuse = bool(video_dit_config.get("fuse_vae_embedding_in_latents", True))
            if user_fuse:
                raise ValueError(
                    "Memory mode requires `video_dit_config['fuse_vae_embedding_in_latents']=False`."
                )
            loader_dit_config["fuse_vae_embedding_in_latents"] = True

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=loader_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        base_video_expert = components.dit
        if memory:
            memory_dit_config = dict(video_dit_config)
            video_expert = MemoryVideoDiT(
                memory=True,
                memory_sizes=memory_sizes,
                memory_kernel_long=memory_kernel_long,
                memory_kernel_mid=memory_kernel_mid,
                memory_kernel_current=memory_kernel_current,
                memory_pred_size=memory_pred_size,
                memory_gain=memory_gain,
                memory_zero_time=memory_zero_time,
                **memory_dit_config,
            ).to(device=device, dtype=torch_dtype)
            video_expert.load_state_dict(base_video_expert.state_dict(), strict=False)

            video_expert._init_memory_patches_from_patch_embedding()
            del base_video_expert
        else:
            video_expert = base_video_expert

        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot_cls = MemoryMoT if memory else MoT
        mot = mot_cls(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            memory=memory,
            memory_sizes=memory_sizes,
            memory_kernel_long=memory_kernel_long,
            memory_kernel_mid=memory_kernel_mid,
            memory_kernel_current=memory_kernel_current,
            memory_pred_size=memory_pred_size,
            memory_gain=memory_gain,
            memory_zero_time=memory_zero_time,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model


    @torch.no_grad()
    def _build_memory_attention_mask(
        self,
        memory_meta: Dict[str, Any],
        action_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:


        L_long = int(memory_meta["L_long"])
        L_mid = int(memory_meta["L_mid"])
        L_curr = int(memory_meta["L_curr"])
        L_pred = int(memory_meta["L_pred"])
        tokens_per_frame_pred = int(memory_meta["tokens_per_frame_pred"])

        L_hist = L_long + L_mid + L_curr
        video_seq_len = L_hist + L_pred
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)


        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_memory_self_attn_mask(
            L_long=L_long,
            L_mid=L_mid,
            L_curr=L_curr,
            L_pred=L_pred,
            tokens_per_frame_pred=tokens_per_frame_pred,
            device=device,
        )


        if action_seq_len > 0:
            mask[video_seq_len:, video_seq_len:] = True
            mask[video_seq_len:, :L_hist] = True
        return mask

    @torch.no_grad()

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[Dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        memory_seq_lens: Optional[Dict[str, int]] = None,
    ) -> torch.Tensor:
        if not self.memory:
            return super()._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )

        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            memory_seq_lens=memory_seq_lens or {},
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()

    @torch.no_grad()

    @torch.no_grad()

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        num_video_inference_steps: Optional[int] = None,
        num_action_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        history_long: Optional[torch.Tensor] = None,
        history_mid: Optional[torch.Tensor] = None,
        history_short: Optional[torch.Tensor] = None,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            num_video_inference_steps=num_video_inference_steps,
            num_action_inference_steps=num_action_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
        )


class Abot06(_Base):


    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.memory:
            raise ValueError(
                "`Abot06` requires `memory=True`; "
                "use `Abot06` for the no-history variant."
            )
    @staticmethod
    def _merge_pred_branches(video_pre_noisy: Dict[str, Any], video_pre_cond: Dict[str, Any]):


        L_hist = int(video_pre_noisy["meta"]["memory"]["L_hist"])
        L_pred = int(video_pre_noisy["meta"]["memory"]["L_pred"])
        if int(video_pre_cond["meta"]["memory"]["L_hist"]) != L_hist:
            raise ValueError("noisy/cond pre_dit produced inconsistent L_hist")
        if int(video_pre_cond["meta"]["memory"]["L_pred"]) != L_pred:
            raise ValueError("noisy/cond pre_dit produced inconsistent L_pred")

        hist_tokens = video_pre_noisy["tokens"][:, :L_hist]
        noisy_tokens = video_pre_noisy["tokens"][:, L_hist:]
        cond_tokens = video_pre_cond["tokens"][:, L_hist:]
        merged_tokens = torch.cat([hist_tokens, noisy_tokens, cond_tokens], dim=1)

        hist_freqs = video_pre_noisy["freqs"][:L_hist]
        noisy_freqs = video_pre_noisy["freqs"][L_hist:]
        cond_freqs = video_pre_cond["freqs"][L_hist:]
        merged_freqs = torch.cat([hist_freqs, noisy_freqs, cond_freqs], dim=0)

        hist_tmod = video_pre_noisy["t_mod"][:, :L_hist]
        noisy_tmod = video_pre_noisy["t_mod"][:, L_hist:]
        cond_tmod = video_pre_cond["t_mod"][:, L_hist:]
        merged_tmod = torch.cat([hist_tmod, noisy_tmod, cond_tmod], dim=1)

        hist_cmask = video_pre_noisy["context_mask"][:, :L_hist]
        noisy_cmask = video_pre_noisy["context_mask"][:, L_hist:]
        cond_cmask = video_pre_cond["context_mask"][:, L_hist:]
        merged_cmask = torch.cat([hist_cmask, noisy_cmask, cond_cmask], dim=1)

        return merged_tokens, merged_freqs, merged_tmod, merged_cmask, L_hist, L_pred

    @torch.no_grad()
    def _build_dual_memory_attention_mask(
        self,
        L_hist: int,
        L_pred: int,
        action_seq_len: int,
        tokens_per_frame_pred: int,
        device: torch.device,
    ) -> torch.Tensor:

        noisy_start = L_hist
        noisy_end = L_hist + L_pred
        cond_start = noisy_end
        cond_end = noisy_end + L_pred
        total_video = cond_end
        total = total_video + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)


        if L_hist > 0:
            mask[:L_hist, :L_hist] = True


        if L_pred > 0:
            if L_hist > 0:
                mask[noisy_start:noisy_end, :L_hist] = True
            mask[noisy_start:noisy_end, noisy_start:noisy_end] = (
                self.video_expert.build_video_to_video_mask(
                    video_seq_len=L_pred,
                    video_tokens_per_frame=tokens_per_frame_pred,
                    device=device,
                )
            )


            if L_hist > 0:
                mask[cond_start:cond_end, :L_hist] = True
            mask[cond_start:cond_end, cond_start:cond_end] = (
                self.video_expert.build_video_to_video_mask(
                    video_seq_len=L_pred,
                    video_tokens_per_frame=tokens_per_frame_pred,
                    device=device,
                )
            )


        if action_seq_len > 0:
            mask[total_video:, total_video:] = True
            if L_hist > 0:
                mask[total_video:, :L_hist] = True
            if L_pred > 0:
                mask[total_video:, cond_start:cond_end] = True

        return mask

    @torch.no_grad()
    def _video_only_forward(
        self,
        video_pre: Dict[str, Any],
        attn_mask_v: torch.Tensor,
    ) -> torch.Tensor:


        x_tokens = video_pre["tokens"]
        for block in self.video_expert.blocks:
            x_tokens = block(
                x_tokens,
                video_pre["context"],
                video_pre["t_mod"],
                video_pre["freqs"],
                context_mask=video_pre["context_mask"],
                self_attn_mask=attn_mask_v,
            )
        return self.video_expert.post_dit(x_tokens, video_pre)

    @torch.no_grad()
    def _build_stage2_attention_mask(
        self,
        memory_meta: Dict[str, Any],
        action_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:


        L_long = int(memory_meta["L_long"])
        L_mid = int(memory_meta["L_mid"])
        L_curr = int(memory_meta["L_curr"])
        L_pred = int(memory_meta["L_pred"])
        tokens_per_frame_pred = int(memory_meta["tokens_per_frame_pred"])
        L_hist = L_long + L_mid + L_curr
        video_seq_len = L_hist + L_pred
        total = video_seq_len + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_memory_self_attn_mask(
            L_long=L_long,
            L_mid=L_mid,
            L_curr=L_curr,
            L_pred=L_pred,
            tokens_per_frame_pred=tokens_per_frame_pred,
            device=device,
        )
        if action_seq_len > 0:
            mask[video_seq_len:, video_seq_len:] = True
            if L_hist > 0:
                mask[video_seq_len:, :L_hist] = True
            mask[video_seq_len:, L_hist:video_seq_len] = True
        return mask

    @torch.no_grad()
    def _denoise_video_with_history(
        self,
        latents_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        history_long: torch.Tensor,
        history_mid: torch.Tensor,
        history_short: torch.Tensor,
        num_inference_steps: int,
        sigma_shift: Optional[float],
    ) -> torch.Tensor:

        infer_timesteps_video, infer_deltas_video = (
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video.dtype, device=self.device
            )
            video_pre = self.video_expert.pre_dit(
                x=latents_video,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=False,
                history_long=history_long,
                history_mid=history_mid,
                history_short=history_short,
            )
            memory_meta = video_pre["meta"]["memory"]
            attn_mask_v = self.video_expert.build_memory_self_attn_mask(
                L_long=int(memory_meta["L_long"]),
                L_mid=int(memory_meta["L_mid"]),
                L_curr=int(memory_meta["L_curr"]),
                L_pred=int(memory_meta["L_pred"]),
                tokens_per_frame_pred=int(memory_meta["tokens_per_frame_pred"]),
                device=video_pre["tokens"].device,
            )
            pred_video = self._video_only_forward(video_pre, attn_mask_v)
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video
            )
        return latents_video

    @torch.no_grad()
    def _denoise_action_with_clean_video(
        self,
        latents_action: torch.Tensor,
        latents_video_clean: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        history_long: torch.Tensor,
        history_mid: torch.Tensor,
        history_short: torch.Tensor,
        num_inference_steps: int,
        sigma_shift: Optional[float],
    ) -> torch.Tensor:

        batch_size = latents_video_clean.shape[0]
        timestep_video_zero = torch.zeros(
            (batch_size,), dtype=latents_video_clean.dtype, device=self.device
        )
        video_pre_cond = self.video_expert.pre_dit(
            x=latents_video_clean,
            timestep=timestep_video_zero,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=False,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
        )
        memory_meta = video_pre_cond["meta"]["memory"]
        L_hist = int(memory_meta["L_hist"])
        L_pred = int(memory_meta["L_pred"])
        video_seq_len = L_hist + L_pred
        action_seq_len = int(latents_action.shape[1])

        attention_mask = self._build_stage2_attention_mask(
            memory_meta=memory_meta,
            action_seq_len=action_seq_len,
            device=video_pre_cond["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre_cond["tokens"],
            video_freqs=video_pre_cond["freqs"],
            video_t_mod=video_pre_cond["t_mod"],
            video_context_payload={
                "context": video_pre_cond["context"],
                "mask": video_pre_cond["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            memory_seq_lens={"video": L_hist, "action": 0},
        )

        infer_timesteps_action, infer_deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device
            )
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                memory_seq_lens={"video": L_hist, "action": 0},
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )
        return latents_action

    @torch.no_grad()
    def _prepare_context(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
    ):
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} "
                    f"and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        return context, context_mask

    @torch.no_grad()
    def infer_from_latents(
        self,
        history_long: torch.Tensor,
        history_mid: torch.Tensor,
        history_short: torch.Tensor,
        action_horizon: int,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        num_video_inference_steps: Optional[int] = None,
        num_action_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        decode_video: bool = True,
    ) -> Dict[str, Any]:
        del text_cfg_scale
        video_steps = int(num_video_inference_steps if num_video_inference_steps is not None else num_inference_steps)
        action_steps = int(num_action_inference_steps if num_action_inference_steps is not None else num_inference_steps)
        if action is not None:
            logger.warning(
                "`Abot06.infer_from_latents` ignores `action`; the model conditions "
                "action on its own predicted clean video, not on a GT action input."
            )

        if history_long.ndim != 5 or history_mid.ndim != 5 or history_short.ndim != 5:
            raise ValueError(
                "All history latents must be 5D `(B, C, T_seg, H_lat, W_lat)`. Got "
                f"long={tuple(history_long.shape)}, mid={tuple(history_mid.shape)}, "
                f"short={tuple(history_short.shape)}"
            )
        long_size, mid_size, curr_size = self.memory_sizes
        if history_long.shape[2] != long_size:
            raise ValueError(f"`history_long` T must be {long_size}, got {history_long.shape[2]}")
        if history_mid.shape[2] != mid_size:
            raise ValueError(f"`history_mid` T must be {mid_size}, got {history_mid.shape[2]}")
        if history_short.shape[2] != curr_size:
            raise ValueError(f"`history_short` T must be {curr_size}, got {history_short.shape[2]}")

        self.eval()
        context, context_mask = self._prepare_context(prompt, context, context_mask, proprio)

        history_long = history_long.to(device=self.device, dtype=self.torch_dtype)
        history_mid = history_mid.to(device=self.device, dtype=self.torch_dtype)
        history_short = history_short.to(device=self.device, dtype=self.torch_dtype)
        batch_size = history_short.shape[0]
        latent_h = history_short.shape[3]
        latent_w = history_short.shape[4]
        latent_t = self.memory_pred_size

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (batch_size, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)


        latents_video = self._denoise_video_with_history(
            latents_video=latents_video,
            context=context,
            context_mask=context_mask,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
            num_inference_steps=video_steps,
            sigma_shift=sigma_shift,
        )

        latents_action = self._denoise_action_with_clean_video(
            latents_action=latents_action,
            latents_video_clean=latents_video,
            context=context,
            context_mask=context_mask,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
            num_inference_steps=action_steps,
            sigma_shift=sigma_shift,
        )

        pred_latents = latents_video.detach()
        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if decode_video:
            decode_latents = torch.cat([history_short, pred_latents], dim=2)
            full_video = self._decode_latents(decode_latents, tiled=tiled)
            decoded_video = full_video[1:]
        else:
            decoded_video = None
        return {
            "pred_latents": pred_latents,
            "video": decoded_video,
            "action": action_out,
        }

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        num_video_inference_steps: Optional[int] = None,
        num_action_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        history_long: Optional[torch.Tensor] = None,
        history_mid: Optional[torch.Tensor] = None,
        history_short: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        del negative_prompt, text_cfg_scale, test_action_with_infer_action
        video_steps = int(num_video_inference_steps if num_video_inference_steps is not None else num_inference_steps)
        action_steps = int(num_action_inference_steps if num_action_inference_steps is not None else num_inference_steps)
        if history_long is None or history_mid is None or history_short is None:
            raise ValueError(
                "`Abot06.infer_joint` requires `history_long/mid/short` "
                "(callers must provide history latents; this implementation does not "
                "auto-roll history)."
            )
        if action is not None:
            logger.warning(
                "`Abot06.infer_joint` ignores `action`; action is conditioned "
                "on the predicted clean video, not on a GT input."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                "`input_image` must be resized before infer; expected multiples of 16 but got "
                f"HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        self.eval()
        context, context_mask = self._prepare_context(prompt, context, context_mask, proprio)

        history_long = history_long.to(device=self.device, dtype=self.torch_dtype)
        history_mid = history_mid.to(device=self.device, dtype=self.torch_dtype)
        history_short = history_short.to(device=self.device, dtype=self.torch_dtype)

        latent_h = history_short.shape[3]
        latent_w = history_short.shape[4]
        latent_t = self.memory_pred_size
        batch_size = history_short.shape[0]

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (batch_size, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        latents_video = self._denoise_video_with_history(
            latents_video=latents_video,
            context=context,
            context_mask=context_mask,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
            num_inference_steps=video_steps,
            sigma_shift=sigma_shift,
        )
        latents_action = self._denoise_action_with_clean_video(
            latents_action=latents_action,
            latents_video_clean=latents_video,
            context=context,
            context_mask=context_mask,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
            num_inference_steps=action_steps,
            sigma_shift=sigma_shift,
        )

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        decode_latents = torch.cat([history_short, latents_video], dim=2)
        return {
            "video": self._decode_latents(decode_latents, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        num_video_inference_steps: Optional[int] = None,
        num_action_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        history_long: Optional[torch.Tensor] = None,
        history_mid: Optional[torch.Tensor] = None,
        history_short: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        del input_image, negative_prompt, text_cfg_scale
        video_steps = int(num_video_inference_steps if num_video_inference_steps is not None else num_inference_steps)
        action_steps = int(num_action_inference_steps if num_action_inference_steps is not None else num_inference_steps)
        if history_long is None or history_mid is None or history_short is None:
            raise ValueError("`Abot06.infer_action` requires `history_long/mid/short`.")

        self.eval()
        context, context_mask = self._prepare_context(prompt, context, context_mask, proprio)

        history_long = history_long.to(device=self.device, dtype=self.torch_dtype)
        history_mid = history_mid.to(device=self.device, dtype=self.torch_dtype)
        history_short = history_short.to(device=self.device, dtype=self.torch_dtype)
        batch_size = history_short.shape[0]
        latent_h = history_short.shape[3]
        latent_w = history_short.shape[4]
        latent_t = self.memory_pred_size

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (batch_size, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        latents_video = self._denoise_video_with_history(
            latents_video=latents_video,
            context=context,
            context_mask=context_mask,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
            num_inference_steps=video_steps,
            sigma_shift=sigma_shift,
        )
        latents_action = self._denoise_action_with_clean_video(
            latents_action=latents_action,
            latents_video_clean=latents_video,
            context=context,
            context_mask=context_mask,
            history_long=history_long,
            history_mid=history_mid,
            history_short=history_short,
            num_inference_steps=action_steps,
            sigma_shift=sigma_shift,
        )

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}


