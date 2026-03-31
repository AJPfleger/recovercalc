import pandas as pd
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
