import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_load(daily: pd.DataFrame):
    ax = daily[["trimp", "ctl", "atl", "tsb"]].plot(
        figsize=(10, 5), secondary_y=["tsb"]
    )
    ax.set_ylabel("TRIMP / load")
    ax.right_ax.set_ylabel("TSB")
    plt.tight_layout()
    plt.show()


def plot_progression(daily: pd.DataFrame, weekly: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(daily.index, daily["ctl_ramp_7"], label="ctl_ramp_7")
    plt.plot(daily.index, daily["ctl_ramp_28"], label="ctl_ramp_28")
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("CTL ramp")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if "distance_km" in weekly.columns and "dist_ramp" in weekly.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(
            pd.to_datetime(weekly["week"]),
            weekly["distance_km"],
            label="distance_km",
        )
        plt.plot(pd.to_datetime(weekly["week"]), weekly["dist_ramp"], label="dist_ramp")
        plt.axhline(0.0, linewidth=1)
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_training_overview(daily: pd.DataFrame, weekly: pd.DataFrame):
    import matplotlib.pyplot as plt

    d = daily.copy()
    w = weekly.copy()

    d.index = pd.to_datetime(d.index)
    if "week" in w.columns:
        w["week"] = pd.to_datetime(w["week"])

    d["trimp_7d"] = d["trimp"].rolling(7, min_periods=1).sum()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)

    # 1) load
    ax = axes[0]
    ax.plot(d.index, d["trimp_7d"], label="TRIMP 7d")
    ax.plot(d.index, d["ctl"], label="CTL")
    ax.plot(d.index, d["atl"], label="ATL")
    ax.set_ylabel("Load")
    ax.set_title("Training Load")
    ax.legend()

    # 2) recovery / TSB with bands
    ax = axes[1]
    ax.axhspan(-40, -20, alpha=0.15)
    ax.axhspan(-20, -5, alpha=0.10)
    ax.axhspan(-5, 5, alpha=0.08)
    ax.axhspan(5, 15, alpha=0.10)
    ax.axhspan(15, 30, alpha=0.12)

    ax.plot(d.index, d["tsb"], label="TSB")
    ax.axhline(-20, linewidth=1)
    ax.axhline(-5, linewidth=1)
    ax.axhline(0, linewidth=1)
    ax.axhline(5, linewidth=1)
    ax.axhline(15, linewidth=1)

    ax.text(d.index[0], -30, "very fatigued", va="center")
    ax.text(d.index[0], -12.5, "training", va="center")
    ax.text(d.index[0], 0, "neutral", va="center")
    ax.text(d.index[0], 10, "fresh", va="center")
    ax.text(d.index[0], 22, "peak", va="center")

    ax.set_ylabel("TSB")
    ax.set_title("Recovery State")
    ax.legend()

    # 3) weekly intensity distribution
    ax = axes[2]
    zone_cols = [c for c in ["z1", "z2", "z3", "z4", "z5"] if c in w.columns]
    if zone_cols:
        bottom = np.zeros(len(w))
        for col in zone_cols:
            ax.bar(w["week"], w[col], bottom=bottom, width=5, label=col.upper())
            bottom += w[col].to_numpy()
    ax.set_ylabel("Fraction")
    ax.set_title("Weekly Intensity Distribution")
    ax.legend()

    plt.tight_layout()
    plt.show()
