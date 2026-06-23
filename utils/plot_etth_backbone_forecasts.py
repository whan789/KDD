import argparse
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_STYLES: Dict[str, Dict[str, str]] = {
    "TimeXer": {"color": "#c46a1a", "label": "TimeXer backbone"},
    "iTransformer": {"color": "#7c4dff", "label": "iTransformer backbone"},
    "DLinear": {"color": "#2c8f8a", "label": "DLinear backbone"},
}


def finite_differences(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d1 = np.zeros_like(values)
    d1[1:] = values[1:] - values[:-1]
    d2 = np.zeros_like(values)
    d2[1:] = d1[1:] - d1[:-1]
    return d1, d2


def infer_results_dir(checkpoint_dir: str) -> str:
    setting = os.path.basename(os.path.normpath(checkpoint_dir))
    return os.path.join("/workspace/results", setting)


def sanitize_data_name(data_path: str) -> str:
    stem = os.path.splitext(os.path.basename(data_path))[0]
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem)


def compute_horizon_from_border(
    csv_path: str,
    sample_idx: int,
    seq_len: int,
    pred_len: int,
    test_border1: int,
) -> Tuple[pd.Timestamp, pd.Timestamp, int, int, pd.DatetimeIndex]:
    df = pd.read_csv(csv_path, usecols=["date"])
    dates = pd.to_datetime(df["date"])
    horizon_start = test_border1 + sample_idx + seq_len
    horizon_end = horizon_start + pred_len - 1
    horizon_dates = pd.DatetimeIndex(dates.iloc[horizon_start:horizon_end + 1])
    return dates.iloc[horizon_start], dates.iloc[horizon_end], horizon_start, horizon_end, horizon_dates


def compute_dataset_horizon(
    dataset_name: str,
    csv_path: str,
    sample_idx: int,
    seq_len: int,
    pred_len: int,
) -> Tuple[pd.Timestamp, pd.Timestamp, int, int, pd.DatetimeIndex]:
    dataset_key = dataset_name.lower()

    if dataset_key in {"etth1", "etth2"}:
        test_border1 = 12 * 30 * 24 + 4 * 30 * 24 - seq_len
        return compute_horizon_from_border(csv_path, sample_idx, seq_len, pred_len, test_border1)

    if dataset_key in {"ettm1", "ettm2"}:
        test_border1 = 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len
        return compute_horizon_from_border(csv_path, sample_idx, seq_len, pred_len, test_border1)

    if dataset_key in {"exchangerate", "weather"}:
        df = pd.read_csv(csv_path)
        num_train = int(len(df) * 0.7)
        num_test = int(len(df) * 0.2)
        test_border1 = len(df) - num_test - seq_len
        return compute_horizon_from_border(csv_path, sample_idx, seq_len, pred_len, test_border1)

    raise ValueError(f"Unsupported dataset for horizon computation: {dataset_name}")


def find_extrema_indices(values: np.ndarray) -> Tuple[List[int], List[int]]:
    peak_indices: List[int] = []
    valley_indices: List[int] = []
    for idx in range(1, len(values) - 1):
        if values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]:
            peak_indices.append(idx)
        if values[idx] <= values[idx - 1] and values[idx] < values[idx + 1]:
            valley_indices.append(idx)
    return peak_indices, valley_indices


def select_landmarks(values: np.ndarray, max_points: int = 3) -> Tuple[List[int], List[int]]:
    peaks, valleys = find_extrema_indices(values)
    peak_sorted = sorted(peaks, key=lambda idx: values[idx], reverse=True)[:max_points]
    valley_sorted = sorted(valleys, key=lambda idx: values[idx])[:max_points]
    return sorted(peak_sorted), sorted(valley_sorted)


def load_series(results_dir: str, sample_idx: int, channel_idx: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.load(os.path.join(results_dir, "pred.npy"))
    true = np.load(os.path.join(results_dir, "true.npy"))
    if channel_idx is None:
        channel_idx = pred.shape[-1] - 1
    return pred[sample_idx, :, channel_idx], true[sample_idx, :, channel_idx]


def resolve_checkpoint_dir(
    model_name: str,
    dataset_key: str,
    data_path: Optional[str],
    seq_len: int,
    pred_len: int,
) -> str:
    base_pattern = (
        "/workspace/checkpoints/long_term_forecast_test_{model}_{dataset}_ftM_sl{seq_len}_ll48_pl{pred_len}_"
        "dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0_seed42_som0"
    )

    candidates = [dataset_key]
    if dataset_key == "custom" and data_path:
        candidates.insert(0, f"custom_{sanitize_data_name(data_path)}")

    for candidate in candidates:
        checkpoint_dir = base_pattern.format(
            model=model_name,
            dataset=candidate,
            seq_len=seq_len,
            pred_len=pred_len,
        )
        if os.path.exists(checkpoint_dir):
            return checkpoint_dir

    # Prefer the new custom-specific path even if it does not exist yet so the
    # caller sees the intended location in error cases.
    preferred = candidates[0]
    return base_pattern.format(
        model=model_name,
        dataset=preferred,
        seq_len=seq_len,
        pred_len=pred_len,
    )


def plot_model_forecast(
    model_name: str,
    checkpoint_dir: str,
    dataset_name: str,
    csv_path: str,
    output_path: str,
    sample_idx: int,
    seq_len: int,
    pred_len: int,
    channel_name: str,
    channel_idx: Optional[int],
) -> None:
    style = MODEL_STYLES[model_name]
    results_dir = infer_results_dir(checkpoint_dir)
    pred, true = load_series(results_dir, sample_idx, channel_idx)
    d1_pred, d2_pred = finite_differences(pred)
    d1_true, d2_true = finite_differences(true)

    start_dt, end_dt, raw_start, raw_end, _ = compute_dataset_horizon(
        dataset_name=dataset_name,
        csv_path=csv_path,
        sample_idx=sample_idx,
        seq_len=seq_len,
        pred_len=pred_len,
    )
    peak_indices, valley_indices = select_landmarks(true, max_points=3)

    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    fig.suptitle(
        f"{dataset_name} {channel_name}: ground truth vs {model_name} backbone forecast | test sample {sample_idx}\n"
        f"Shared {pred_len}-step interval: {start_dt:%Y-%m-%d %H:%M} to {end_dt:%Y-%m-%d %H:%M}",
        fontsize=22,
        y=0.98,
    )

    x = np.arange(pred_len)
    peak_color = "#ff5b5b"
    valley_color = "#3b82f6"

    panels = [
        (true, pred, "Raw values", channel_name),
        (d1_true, d1_pred, "First difference: d1[t] = value[t] - value[t-1]", "d1"),
        (d2_true, d2_pred, "Second difference: d2[t] = d1[t] - d1[t-1]", "d2"),
    ]

    for axis, (true_values, pred_values, title, ylabel) in zip(axes, panels):
        axis.plot(x, true_values, color="#1f2937", linewidth=2.2, label="Ground truth")
        axis.plot(x, pred_values, color=style["color"], linewidth=2.0, label=style["label"])
        axis.axhline(0.0, color="#9ca3af", linewidth=0.9)
        axis.grid(True, axis="y", alpha=0.22)
        for step in range(pred_len):
            axis.axvline(step, color="#cbd5e1", linestyle=":", linewidth=0.6, alpha=0.45, zorder=0)
        axis.set_ylabel(ylabel, fontsize=12)
        axis.set_title(title, fontsize=17, loc="left")
        for idx in peak_indices:
            axis.axvline(idx, color=peak_color, linestyle="--", linewidth=1.0, alpha=0.18)
        for idx in valley_indices:
            axis.axvline(idx, color=valley_color, linestyle="--", linewidth=1.0, alpha=0.18)
        axis.legend(loc="upper right", fontsize=12)

    axes[0].scatter(peak_indices, true[peak_indices], color=peak_color, s=28, zorder=5)
    axes[0].scatter(valley_indices, true[valley_indices], color=valley_color, s=28, marker="v", zorder=5)
    axes[2].set_xlabel(f"forecast step in shared {pred_len}-step interval", fontsize=12)

    fig.text(
        0.01,
        0.012,
        f"Source: {dataset_name} {channel_name} {model_name} seed42, test sample {sample_idx}, "
        f"inverse-scaled | raw indices [{raw_start}:{raw_end + 1}]",
        fontsize=11,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ETTh backbone forecast diagnostics from saved predictions.")
    parser.add_argument("--dataset", type=str, default="ETTh2")
    parser.add_argument("--data-key", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--root-path", type=str, default=None)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--channel-name", type=str, default="OT")
    parser.add_argument("--channel-idx", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="/workspace/analysis/forecast_plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_key = args.data_key or args.dataset

    if args.root_path is not None and args.data_path is not None:
        csv_path = os.path.join(args.root_path, args.data_path)
    else:
        csv_defaults = {
            "ETTh1": ("/workspace/datasets/ETT-small", "ETTh1.csv"),
            "ETTh2": ("/workspace/datasets/ETT-small", "ETTh2.csv"),
            "ETTm1": ("/workspace/datasets/ETT-small", "ETTm1.csv"),
            "ETTm2": ("/workspace/datasets/ETT-small", "ETTm2.csv"),
            "ExchangeRate": ("/workspace/datasets/exchange_rate", "exchange_rate.csv"),
            "Weather": ("/workspace/datasets/weather", "weather.csv"),
        }
        root_path, data_path = csv_defaults[args.dataset]
        csv_path = os.path.join(root_path, data_path)

    checkpoint_dirs = {
        model_name: resolve_checkpoint_dir(
            model_name=model_name,
            dataset_key=dataset_key,
            data_path=args.data_path,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
        )
        for model_name in ("TimeXer", "iTransformer", "DLinear")
    }

    for model_name, checkpoint_dir in checkpoint_dirs.items():
        output_path = os.path.join(args.output_dir, f"{args.dataset}_{model_name}_sample{args.sample_idx}.png")
        plot_model_forecast(
            model_name=model_name,
            checkpoint_dir=checkpoint_dir,
            dataset_name=args.dataset,
            csv_path=csv_path,
            output_path=output_path,
            sample_idx=args.sample_idx,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            channel_name=args.channel_name,
            channel_idx=args.channel_idx,
        )
        print(output_path)


if __name__ == "__main__":
    main()
