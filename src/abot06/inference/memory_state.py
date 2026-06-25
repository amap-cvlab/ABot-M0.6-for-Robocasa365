from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryConfig:


    L: int
    M: int
    S: int
    P: int
    r: int
    num_actions: int
    V: int
    N_future: int
    Z: int
    temporal_factor: int
    z_dim: int
    latent_h: int
    latent_w: int

    @classmethod
    def from_cfg_and_model(cls, cfg: DictConfig, model) -> "MemoryConfig":


        memory_sizes = cfg.model.get("memory_sizes")
        if memory_sizes is None:
            raise ValueError(
                "`cfg.model.memory_sizes` is required for Memory rollout. "
                "Make sure you are using a `abot06` model config."
            )
        if isinstance(memory_sizes, (DictConfig, ListConfig)):
            memory_sizes = OmegaConf.to_container(memory_sizes, resolve=True)
        memory_sizes = list(memory_sizes)
        if len(memory_sizes) != 3:
            raise ValueError(f"`memory_sizes` must be length 3 (L, M, S), got {memory_sizes}")
        L, M, S = int(memory_sizes[0]), int(memory_sizes[1]), int(memory_sizes[2])
        if S != 1:
            raise ValueError(
                f"Memory rollout currently requires S (current/short) == 1, got S={S}. "
                "The model and dataset only support S=1."
            )

        P = int(cfg.model.get("memory_pred_size", 2))


        data_train = cfg.data.train
        num_frames = int(data_train.num_frames)
        r = int(data_train.action_video_freq_ratio)
        num_actions = num_frames - 1

        if num_actions <= 0:
            raise ValueError(f"`num_frames` must be >= 2, got {num_frames}")
        if r <= 0:
            raise ValueError(f"`action_video_freq_ratio` must be positive, got {r}")
        if num_actions % r != 0:
            raise ValueError(
                f"`num_actions` ({num_actions} = num_frames-1) must be divisible by "
                f"`action_video_freq_ratio` ({r}), but {num_actions} % {r} = {num_actions % r}."
            )

        N_future = num_actions // r
        V = N_future + 1


        temporal_factor = int(model.vae.temporal_downsample_factor)
        z_dim = int(model.vae.model.z_dim)
        upsampling_factor = int(model.vae.upsampling_factor)


        if (V - 1) % temporal_factor != 0:
            raise ValueError(
                f"Video frames per chunk V={V} is incompatible with VAE "
                f"temporal_downsample_factor={temporal_factor}: "
                f"(V-1)={V - 1} is not divisible by {temporal_factor}."
            )
        chunk_latent_t = (V - 1) // temporal_factor + 1
        if chunk_latent_t != P + 1:
            raise ValueError(
                f"VAE encodes V={V} pixel frames into {chunk_latent_t} latent frames, "
                f"but expected P+1={P + 1} (memory_pred_size={P}). Check num_frames / "
                f"action_video_freq_ratio / memory_pred_size consistency."
            )


        Z_cfg = data_train.get("zero_pad_len", None)
        if Z_cfg is not None:
            Z = int(Z_cfg)
        else:

            Z = L + M
            logger.warning(
                "`zero_pad_len` not found in cfg.data.train; using L+M=%d as fallback.", Z
            )


        if L + M + S != Z + 1:
            logger.warning(
                "L+M+S=%d != Z+1=%d (zero_pad_len=%d). This may cause "
                "train/eval distribution mismatch. Please double-check configs.",
                L + M + S, Z + 1, Z,
            )


        video_size = data_train.get("video_size", [256, 768])
        if isinstance(video_size, (DictConfig, ListConfig)):
            video_size = OmegaConf.to_container(video_size, resolve=True)
        pixel_h, pixel_w = int(video_size[0]), int(video_size[1])
        latent_h = pixel_h // upsampling_factor
        latent_w = pixel_w // upsampling_factor

        result = cls(
            L=L, M=M, S=S, P=P, r=r,
            num_actions=num_actions,
            V=V, N_future=N_future, Z=Z,
            temporal_factor=temporal_factor,
            z_dim=z_dim,
            latent_h=latent_h, latent_w=latent_w,
        )
        logger.info(
            "MemoryConfig: L=%d M=%d S=%d P=%d r=%d num_actions=%d V=%d "
            "N_future=%d Z=%d temporal_factor=%d z_dim=%d latent=(%d,%d)",
            L, M, S, P, r, num_actions, V, N_future, Z,
            temporal_factor, z_dim, latent_h, latent_w,
        )
        return result


class MemoryState:


    def __init__(
        self,
        model,
        memory_cfg: MemoryConfig,
        device,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.cfg = memory_cfg
        self.device = device
        self.dtype = dtype


        self._timeline_lat: torch.Tensor = torch.zeros(
            (1, memory_cfg.z_dim, memory_cfg.Z, memory_cfg.latent_h, memory_cfg.latent_w),
            device=device, dtype=dtype,
        )


        self._last_obs_pixel: Optional[torch.Tensor] = None
        self._last_short_latent: Optional[torch.Tensor] = None


        self._chunk_count: int = 0

        logger.info(
            "MemoryState initialized: Z=%d, P+1=%d, V=%d, timeline shape=%s",
            memory_cfg.Z, memory_cfg.P + 1, memory_cfg.V, tuple(self._timeline_lat.shape),
        )


    def reset(self) -> None:

        self._timeline_lat = torch.zeros(
            (1, self.cfg.z_dim, self.cfg.Z, self.cfg.latent_h, self.cfg.latent_w),
            device=self.device, dtype=self.dtype,
        )
        self._last_obs_pixel = None
        self._last_short_latent = None
        self._chunk_count = 0

    @torch.no_grad()
    def set_observation_rgb(self, image_pixel: torch.Tensor) -> None:


        if image_pixel.ndim != 4 or image_pixel.shape[0] != 1 or image_pixel.shape[1] != 3:
            raise ValueError(
                f"`image_pixel` must be (1, 3, H, W), got {tuple(image_pixel.shape)}"
            )
        self._last_obs_pixel = image_pixel.to(device=self.device, dtype=self.dtype)
        self._last_short_latent = self.model._encode_input_image_latents_tensor(
            self._last_obs_pixel
        )

    def build_histories(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


        if self._last_short_latent is None:
            raise RuntimeError(
                "Call `set_observation_rgb()` before `build_histories()`."
            )

        L, M, S = self.cfg.L, self.cfg.M, self.cfg.S
        total_needed = L + M + S
        T_cur = self._timeline_lat.shape[2]

        if T_cur < total_needed:

            pad = torch.zeros(
                (1, self.cfg.z_dim, total_needed - T_cur,
                 self.cfg.latent_h, self.cfg.latent_w),
                device=self.device, dtype=self.dtype,
            )
            window = torch.cat([pad, self._timeline_lat], dim=2)
        else:
            window = self._timeline_lat[:, :, -total_needed:]

        history_long = window[:, :, :L].contiguous()
        history_mid = window[:, :, L:L + M].contiguous()

        history_short = self._last_short_latent.contiguous()

        return history_long, history_mid, history_short

    def is_first_chunk(self) -> bool:

        return self._chunk_count == 0

    def should_skip_env_fetch_obs(self) -> bool:


        return not self.is_first_chunk()

    @torch.no_grad()
    def after_chunk_executed(self, future_rgb_pixels: torch.Tensor) -> None:


        if self._last_obs_pixel is None:
            raise RuntimeError(
                "Call `set_observation_rgb()` before `after_chunk_executed()`."
            )

        N_future = self.cfg.N_future
        V = self.cfg.V
        P_plus_1 = self.cfg.P + 1


        if future_rgb_pixels.ndim == 4:

            future_rgb_pixels = future_rgb_pixels.unsqueeze(0)
        if future_rgb_pixels.ndim != 5:
            raise ValueError(
                f"`future_rgb_pixels` must be 5D (1, 3, N_future, H, W) or "
                f"4D (N_future, 3, H, W), got {tuple(future_rgb_pixels.shape)}"
            )
        if future_rgb_pixels.shape[2] != N_future:
            raise ValueError(
                f"Expected N_future={N_future} future frames, "
                f"got {future_rgb_pixels.shape[2]}."
            )

        future_rgb_pixels = future_rgb_pixels.to(device=self.device, dtype=self.dtype)


        obs_5d = self._last_obs_pixel.unsqueeze(2)
        video_pixels = torch.cat([obs_5d, future_rgb_pixels], dim=2)

        if video_pixels.shape[2] != V:
            raise ValueError(
                f"Concatenated video has {video_pixels.shape[2]} frames, expected V={V}."
            )


        video_for_vae = video_pixels.squeeze(0)
        new_latents = self.model._encode_video_latents([video_for_vae])

        if new_latents.ndim == 4:
            new_latents = new_latents.unsqueeze(0)


        actual_t = new_latents.shape[2]
        if actual_t != P_plus_1:
            raise ValueError(
                f"VAE encoded {V} pixel frames into {actual_t} latent frames, "
                f"expected P+1={P_plus_1}."
            )


        new_latents = new_latents.to(device=self.device, dtype=self.dtype)
        self._timeline_lat = torch.cat(
            [self._timeline_lat, new_latents], dim=2
        )


        max_len = max(self.cfg.Z + 100 * P_plus_1, self.cfg.L + self.cfg.M + self.cfg.S)
        if self._timeline_lat.shape[2] > max_len:
            self._timeline_lat = self._timeline_lat[:, :, -max_len:]

        self._chunk_count += 1
        logger.debug(
            "MemoryState: chunk %d done, timeline T=%d",
            self._chunk_count, self._timeline_lat.shape[2],
        )


    @staticmethod
    def validate_replan_steps(replan_steps: int, action_video_freq_ratio: int) -> None:


        if replan_steps % action_video_freq_ratio != 0:
            r = action_video_freq_ratio
            raise ValueError(
                f"`replan_steps` ({replan_steps}) must be divisible by "
                f"`action_video_freq_ratio` ({r}) for Memory rollout, "
                f"because we need to collect exactly replan_steps/r future "
                f"observations. Please set replan_steps to a multiple of {r} "
                f"(e.g. {r * (replan_steps // r)} or {r * (replan_steps // r + 1)})."
            )
