import argparse
import json
import os
from collections import defaultdict

import pandas as pd


DEFAULT_GROUP_ORDER = ["atomic_seen", "composite_seen", "composite_unseen"]


def format_time(seconds: float) -> str:

    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds:02d}s"
    if seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:02d}m{remaining_seconds:02d}s"
    hours = seconds // 3600
    remaining = seconds % 3600
    minutes = remaining // 60
    remaining_seconds = remaining % 60
    return f"{hours:02d}h{minutes:02d}m{remaining_seconds:02d}s"


def _parse_result_filename(filename: str) -> tuple[str, str] | None:

    if not filename.startswith("gpu") or not filename.endswith("_results.json"):
        return None
    body = filename[len("gpu"):-len("_results.json")]
    if "_" not in body:
        return None
    gpu_id, env_name = body.split("_", 1)
    return gpu_id, env_name


def summarize_results(output_dir: str) -> None:

    group_stats = defaultdict(lambda: {
        "total_tasks": 0,
        "total_trials": 0,
        "total_successes": 0,
        "total_time": 0.0,
        "max_time": 0.0,
        "psnr_sum": 0.0,
        "psnr_count": 0,
        "success_rates": [],
    })
    task_results: dict[str, dict] = {}
    has_psnr_metric = False

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output dir not found: {output_dir}")

    for group_name in sorted(os.listdir(output_dir)):
        group_dir = os.path.join(output_dir, group_name)
        if not os.path.isdir(group_dir):
            continue

        found_result = False
        for filename in os.listdir(group_dir):
            parsed = _parse_result_filename(filename)
            if parsed is None:
                continue
            _, env_name = parsed
            json_path = os.path.join(group_dir, filename)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
            except Exception as exc:
                print(f"[warn] failed to parse {json_path}: {exc}")
                continue
            found_result = True

            total_episodes = int(result.get("total_episodes", 0))
            successes = int(result.get("successes", 0))
            duration = float(result.get("duration", 0.0))
            success_rate = float(result.get("success_rate", (successes / total_episodes) if total_episodes > 0 else 0.0))

            stats = group_stats[group_name]
            stats["total_tasks"] += 1
            stats["total_trials"] += total_episodes
            stats["total_successes"] += successes
            stats["total_time"] += duration
            stats["max_time"] = max(stats["max_time"], duration)
            stats["success_rates"].append(success_rate)
            if "future_video_psnr_mean" in result:
                has_psnr_metric = True
                if result["future_video_psnr_mean"] is not None:
                    stats["psnr_sum"] += float(result["future_video_psnr_mean"])
                    stats["psnr_count"] += 1

            task_key = f"{group_name}/{env_name}"
            task_result = {
                "group": group_name,
                "env_name": env_name,
                "success_rate": success_rate * 100.0,
                "duration": duration,
                "total_episodes": total_episodes,
                "successes": successes,
                "task_description": result.get("task_description") or "",
            }
            if "future_video_psnr_mean" in result:
                task_result["future_video_psnr_mean"] = (
                    float(result["future_video_psnr_mean"]) if result["future_video_psnr_mean"] is not None else None
                )
            task_results[task_key] = task_result

        if not found_result:

            group_stats.pop(group_name, None)

    if not group_stats:
        print(f"No evaluation results found under {output_dir}")
        return

    print("\n=== RoboCasa Evaluation Results Summary ===")
    print("\nPer-group statistics:")

    ordered_groups = [g for g in DEFAULT_GROUP_ORDER if g in group_stats]
    ordered_groups.extend(g for g in sorted(group_stats.keys()) if g not in ordered_groups)

    df_rows = {
        "Group": [],
        "Avg Task Success Rate (%)": [],
        "Micro Success Rate (%)": [],
        "Num Tasks": [],
        "Total Trials": [],
        "Total Time": [],
        "Max Task Time": [],
    }
    if has_psnr_metric:
        df_rows["Average Future PSNR (dB)"] = []

    overall_task_rates: list[float] = []
    overall_trials = 0
    overall_successes = 0
    overall_time = 0.0
    overall_max_time = 0.0
    overall_psnr_sum = 0.0
    overall_psnr_count = 0

    for group_name in ordered_groups:
        stats = group_stats[group_name]
        macro_rate = (sum(stats["success_rates"]) / len(stats["success_rates"]) * 100.0) if stats["success_rates"] else 0.0
        micro_rate = (stats["total_successes"] / stats["total_trials"] * 100.0) if stats["total_trials"] > 0 else 0.0
        avg_psnr = (stats["psnr_sum"] / stats["psnr_count"]) if stats["psnr_count"] > 0 else None

        print(f"\n{group_name}:")
        print(f"- Tasks evaluated: {stats['total_tasks']}")
        print(f"- Total trials: {stats['total_trials']}")
        print(f"- Total successes: {stats['total_successes']}")
        print(f"- Macro success rate (avg over tasks): {macro_rate:.2f}%")
        print(f"- Micro success rate (trial-weighted): {micro_rate:.2f}%")
        print(f"- Total time: {format_time(stats['total_time'])}")
        print(f"- Max task time: {format_time(stats['max_time'])}")
        if has_psnr_metric:
            if avg_psnr is not None:
                print(f"- Average future-video PSNR: {avg_psnr:.4f} dB")
            else:
                print("- Average future-video PSNR: N/A")

        df_rows["Group"].append(group_name)
        df_rows["Avg Task Success Rate (%)"].append(f"{macro_rate:.2f}")
        df_rows["Micro Success Rate (%)"].append(f"{micro_rate:.2f}")
        df_rows["Num Tasks"].append(stats["total_tasks"])
        df_rows["Total Trials"].append(stats["total_trials"])
        df_rows["Total Time"].append(format_time(stats["total_time"]))
        df_rows["Max Task Time"].append(format_time(stats["max_time"]))
        if has_psnr_metric:
            df_rows["Average Future PSNR (dB)"].append(f"{avg_psnr:.4f}" if avg_psnr is not None else "N/A")

        overall_task_rates.extend(stats["success_rates"])
        overall_trials += stats["total_trials"]
        overall_successes += stats["total_successes"]
        overall_time += stats["total_time"]
        overall_max_time = max(overall_max_time, stats["max_time"])
        overall_psnr_sum += stats["psnr_sum"]
        overall_psnr_count += stats["psnr_count"]

    overall_macro = (sum(overall_task_rates) / len(overall_task_rates) * 100.0) if overall_task_rates else 0.0
    overall_micro = (overall_successes / overall_trials * 100.0) if overall_trials > 0 else 0.0
    overall_psnr = (overall_psnr_sum / overall_psnr_count) if overall_psnr_count > 0 else None

    print("\nOverall:")
    print(f"- Macro success rate: {overall_macro:.2f}%")
    print(f"- Micro success rate: {overall_micro:.2f}%")
    print(f"- Total time: {format_time(overall_time)}")
    print(f"- Longest task time: {format_time(overall_max_time)}")
    if has_psnr_metric:
        if overall_psnr is not None:
            print(f"- Average future-video PSNR: {overall_psnr:.4f} dB")
        else:
            print("- Average future-video PSNR: N/A")

    df_rows["Group"].append("Overall")
    df_rows["Avg Task Success Rate (%)"].append(f"{overall_macro:.2f}")
    df_rows["Micro Success Rate (%)"].append(f"{overall_micro:.2f}")
    df_rows["Num Tasks"].append(sum(s["total_tasks"] for s in group_stats.values()))
    df_rows["Total Trials"].append(overall_trials)
    df_rows["Total Time"].append(format_time(overall_time))
    df_rows["Max Task Time"].append(format_time(overall_max_time))
    if has_psnr_metric:
        df_rows["Average Future PSNR (dB)"].append(f"{overall_psnr:.4f}" if overall_psnr is not None else "N/A")

    ckpt_path = os.environ.get("CKPT", "")
    title = os.path.basename(ckpt_path) if ckpt_path else "Results"

    df = pd.DataFrame(df_rows).set_index("Group").T
    summary_csv_path = os.path.join(output_dir, "summary.csv")
    with open(summary_csv_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        df.to_csv(f)

    task_rows = {
        "Group": [],
        "Env": [],
        "Description": [],
        "Trials": [],
        "Successes": [],
        "Success Rate (%)": [],
        "Duration": [],
    }
    if has_psnr_metric:
        task_rows["Future PSNR (dB)"] = []

    sorted_tasks = sorted(
        task_results.values(),
        key=lambda r: (DEFAULT_GROUP_ORDER.index(r["group"]) if r["group"] in DEFAULT_GROUP_ORDER else 99, r["env_name"]),
    )
    for item in sorted_tasks:
        task_rows["Group"].append(item["group"])
        task_rows["Env"].append(item["env_name"])
        task_rows["Description"].append(item["task_description"])
        task_rows["Trials"].append(item["total_episodes"])
        task_rows["Successes"].append(item["successes"])
        task_rows["Success Rate (%)"].append(f"{item['success_rate']:.2f}")
        task_rows["Duration"].append(format_time(item["duration"]))
        if has_psnr_metric:
            psnr = item.get("future_video_psnr_mean")
            task_rows["Future PSNR (dB)"].append(f"{psnr:.4f}" if psnr is not None else "N/A")

    task_df = pd.DataFrame(task_rows)
    task_csv_path = os.path.join(output_dir, "task_success_rates.csv")
    task_df.to_csv(task_csv_path, index=False)

    summary_json_path = os.path.join(output_dir, "summary.json")
    group_stats_output = {}
    for group_name, stats in group_stats.items():
        macro_rate = (sum(stats["success_rates"]) / len(stats["success_rates"])) if stats["success_rates"] else 0.0
        micro_rate = (stats["total_successes"] / stats["total_trials"]) if stats["total_trials"] > 0 else 0.0
        group_stats_output[group_name] = {
            "total_tasks": stats["total_tasks"],
            "total_trials": stats["total_trials"],
            "total_successes": stats["total_successes"],
            "total_time": stats["total_time"],
            "max_time": stats["max_time"],
            "macro_success_rate": macro_rate,
            "micro_success_rate": micro_rate,
        }
        if has_psnr_metric:
            group_stats_output[group_name]["average_future_video_psnr"] = (
                stats["psnr_sum"] / stats["psnr_count"] if stats["psnr_count"] > 0 else None
            )

    overall_stats_output = {
        "macro_success_rate": overall_macro / 100.0,
        "micro_success_rate": overall_micro / 100.0,
        "total_time": overall_time,
        "max_task_time": overall_max_time,
    }
    if has_psnr_metric:
        overall_stats_output["average_future_video_psnr"] = overall_psnr

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": os.path.basename(os.path.abspath(output_dir)),
                "ckpt": ckpt_path,
                "config": os.environ.get("CONFIG", ""),
                "group_stats": group_stats_output,
                "task_results": task_results,
                "overall": overall_stats_output,
            },
            f,
            indent=4,
        )

    print(f"\n=== Run Information ===")
    print(f"Run ID: {os.path.basename(os.path.abspath(output_dir))}")
    print(f"Results directory: {output_dir}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Task success rates CSV: {task_csv_path}")

    print("\n=== Task Success Rates ===")
    print(task_df.to_string(index=False))

    print("\n=== Results Table ===")
    print(df.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory containing evaluation results")
    args = parser.parse_args()
    summarize_results(args.output_dir)


if __name__ == "__main__":
    main()
