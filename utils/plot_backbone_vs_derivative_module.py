import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def finite_differences(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d1 = np.zeros_like(values)
    d1[1:] = values[1:] - values[:-1]
    d2 = np.zeros_like(values)
    d2[1:] = d1[1:] - d1[:-1]
    return d1, d2


def find_extrema_indices(values: np.ndarray) -> Tuple[List[int], List[int]]:
    peaks: List[int] = []
    valleys: List[int] = []
    for idx in range(1, len(values) - 1):
        if values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]:
            peaks.append(idx)
        if values[idx] <= values[idx - 1] and values[idx] < values[idx + 1]:
            valleys.append(idx)
    return peaks, valleys


def select_landmarks(values: np.ndarray, max_points: int = 5) -> Tuple[List[int], List[int]]:
    peaks, valleys = find_extrema_indices(values)
    peaks = sorted(sorted(peaks, key=lambda idx: values[idx], reverse=True)[:max_points])
    valleys = sorted(sorted(valleys, key=lambda idx: values[idx])[:max_points])
    return peaks, valleys


def main() -> None:
    dataset = "ETTh1"
    model = "iTransformer"
    pred_len = 720
    seed = 42
    sample_idx = 0
    channel_idx = -1
    channel_name = "OT"

    base_setting = (
        f"long_term_forecast_test_{model}_{dataset}_ftM_sl96_ll48_pl{pred_len}_"
        f"dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0_seed{seed}"
    )
    backbone_dir = os.path.join("/workspace/results", f"{base_setting}_som0")
    module_dir = os.path.join("/workspace/results", f"{base_setting}_som1_derivative_sharpening")

    backbone_pred = np.load(os.path.join(backbone_dir, "pred.npy"))[sample_idx, :, channel_idx]
    true = np.load(os.path.join(backbone_dir, "true.npy"))[sample_idx, :, channel_idx]
    module_pred = np.load(os.path.join(module_dir, "pred.npy"))[sample_idx, :, channel_idx]

    true_d1, true_d2 = finite_differences(true)
    backbone_d1, backbone_d2 = finite_differences(backbone_pred)
    module_d1, module_d2 = finite_differences(module_pred)
    correction = module_pred - backbone_pred

    peak_indices, valley_indices = select_landmarks(true, max_points=5)
    x = np.arange(pred_len)

    output_dir = "/workspace/analysis/forecast_plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ETTh1_iTransformer_pl720_backbone_vs_d1d2_sample0.png")

    fig, axes = plt.subplots(4, 1, figsize=(24, 14), sharex=True)
    fig.suptitle(
        f"{dataset} {channel_name}: ground truth vs {model} backbone vs backbone+d1/d2 | "
        f"pred_len {pred_len}, seed {seed}, test sample {sample_idx}",
        fontsize=22,
        y=0.98,
    )

    panels = [
        (true, backbone_pred, module_pred, "Raw values", channel_name),
        (true_d1, backbone_d1, module_d1, "First difference: d1[t] = value[t] - value[t-1]", "d1"),
        (true_d2, backbone_d2, module_d2, "Second difference: d2[t] = d1[t] - d1[t-1]", "d2"),
    ]

    for axis, (truth, backbone, module, title, ylabel) in zip(axes[:3], panels):
        axis.plot(x, truth, color="#111827", linewidth=2.0, label="Ground truth")
        axis.plot(x, backbone, color="#7c4dff", linewidth=1.6, label="iTransformer backbone")
        axis.plot(x, module, color="#d97706", linewidth=1.6, label="backbone+d1/d2 module")
        axis.axhline(0.0, color="#9ca3af", linewidth=0.8)
        axis.grid(True, axis="y", alpha=0.22)
        for step in range(pred_len):
            axis.axvline(step, color="#cbd5e1", linestyle=":", linewidth=0.35, alpha=0.22, zorder=0)
        for idx in peak_indices:
            axis.axvline(idx, color="#ff5b5b", linestyle="--", linewidth=1.0, alpha=0.2)
        for idx in valley_indices:
            axis.axvline(idx, color="#3b82f6", linestyle="--", linewidth=1.0, alpha=0.2)
        axis.set_title(title, fontsize=16, loc="left")
        axis.set_ylabel(ylabel)
        axis.legend(loc="upper right")

    axes[0].scatter(peak_indices, true[peak_indices], color="#ff5b5b", s=22, zorder=5)
    axes[0].scatter(valley_indices, true[valley_indices], color="#3b82f6", s=22, marker="v", zorder=5)

    axes[3].plot(x, correction, color="#b45309", linewidth=1.4, label="module correction")
    axes[3].axhline(0.0, color="#9ca3af", linewidth=0.8)
    axes[3].grid(True, axis="y", alpha=0.22)
    for step in range(pred_len):
        axes[3].axvline(step, color="#cbd5e1", linestyle=":", linewidth=0.35, alpha=0.22, zorder=0)
    for idx in peak_indices:
        axes[3].axvline(idx, color="#ff5b5b", linestyle="--", linewidth=1.0, alpha=0.2)
    for idx in valley_indices:
        axes[3].axvline(idx, color="#3b82f6", linestyle="--", linewidth=1.0, alpha=0.2)
    axes[3].set_title("Correction: backbone+d1/d2 - backbone", fontsize=16, loc="left")
    axes[3].set_ylabel("delta")
    axes[3].set_xlabel(f"forecast step in shared {pred_len}-step interval")
    axes[3].legend(loc="upper right")

    fig.text(
        0.01,
        0.012,
        "Source: saved pred.npy/true.npy from results folders; values are in saved result scale.",
        fontsize=11,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
