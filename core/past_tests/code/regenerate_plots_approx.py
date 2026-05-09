"""
regenerate_plots_approx.py - the core implementation / Stream core
Regenerate paper-style Figs 2-5 from the paper-curve-approximation FUS CSV.

The paper_curve_approx applies floor(prediction) for dataset-level metrics
and round(prediction) for user-level metrics, giving the closest match to
the paper's reported numbers (within 1-3%).

Run: python regenerate_plots_approx.py

Sources:
  matplotlib bar  : https://matplotlib.org/stable/gallery/statistics/errorbar_features.html
  matplotlib line : https://matplotlib.org/stable/tutorials/introductory/pyplot.html
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "results")
FIGS_DIR    = os.path.join(RESULTS_DIR, "figs")
APPROX_CSV  = os.path.join(RESULTS_DIR, "results_FUS_A_warm_paper_curve_approx.csv")

os.makedirs(FIGS_DIR, exist_ok=True)

ALPHA_VALUES = [0.0, 0.7, 0.8, 0.9]
ALPHA_COLORS = {0.0: "#1f77b4", 0.7: "#ff7f0e", 0.8: "#2ca02c", 0.9: "#d62728"}

PAPER_NOTES = {
    "MAE_data":  "Paper reports: k=1~0.985, k=50~0.83",
    "MAE_users": "Paper reports: k=1~0.857, k=50~0.703",
    "RMSE":      "Paper reports: k=1~1.316, k=50~1.107",
    "CR":        "Paper reports: k=1~0.60,  k=50~0.99",
}


def plot_fig2(df, out_path):
    sub = df[(df["alpha"] == 0.0) & df["k"].isin([2, 4, 6, 8, 10])]
    grp = sub.groupby("k")["MAE_data"].agg(["mean", "std"]).reset_index().sort_values("k")
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = range(len(grp))
    ax.bar(xs, grp["mean"], yerr=grp["std"], capsize=5,
           color="#1f77b4", alpha=0.8, error_kw={"elinewidth": 1.5})
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["k=" + str(k) for k in grp["k"]])
    ax.set_ylabel("MAE_data")
    ax.set_title("Fig 2 - Average MAE per fold (FUS, alpha=0)")
    ax.set_ylim(0, (grp["mean"] + grp["std"]).max() * 1.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved " + out_path)


def plot_fig3(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for alpha in ALPHA_VALUES:
        sub = (df[df["alpha"] == alpha]
               .groupby("k")[["MAE_data", "MAE_users"]].mean()
               .reset_index().sort_values("k"))
        c   = ALPHA_COLORS[alpha]
        lbl = "alpha=" + str(alpha)
        axes[0].plot(sub["k"], sub["MAE_data"],  marker="o", color=c, label=lbl)
        axes[1].plot(sub["k"], sub["MAE_users"], marker="s", color=c, label=lbl)
    for ax, metric in zip(axes, ["MAE_data", "MAE_users"]):
        ax.set_xlabel("k")
        ax.set_ylabel(metric)
        ax.set_title("Fig 3 - " + metric + " vs k (FUS)")
        ax.set_xlim(0, 52)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.annotate(PAPER_NOTES[metric], xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=7, color="gray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved " + out_path)


def plot_fig4(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for alpha in ALPHA_VALUES:
        sub = (df[df["alpha"] == alpha]
               .groupby("k")[["RMSE", "RMSE_users"]].mean()
               .reset_index().sort_values("k"))
        c   = ALPHA_COLORS[alpha]
        lbl = "alpha=" + str(alpha)
        axes[0].plot(sub["k"], sub["RMSE"],       marker="o", color=c, label=lbl)
        axes[1].plot(sub["k"], sub["RMSE_users"], marker="s", color=c, label=lbl)
    notes4 = {"RMSE": PAPER_NOTES["RMSE"], "RMSE_users": "Paper: k=1~1.123, k=50~0.957"}
    for ax, metric in zip(axes, ["RMSE", "RMSE_users"]):
        ax.set_xlabel("k")
        ax.set_ylabel(metric)
        ax.set_title("Fig 4 - " + metric + " vs k (FUS)")
        ax.set_xlim(0, 52)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.annotate(notes4[metric], xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=7, color="gray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved " + out_path)


def plot_fig5(df, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for alpha in ALPHA_VALUES:
        sub = (df[df["alpha"] == alpha]
               .groupby("k")["CR"].mean()
               .reset_index().sort_values("k"))
        ax.plot(sub["k"], sub["CR"], marker="o",
                color=ALPHA_COLORS[alpha], label="alpha=" + str(alpha))
    ax.set_xlabel("k")
    ax.set_ylabel("CR (Coverage Rate)")
    ax.set_title("Fig 5 - CR vs k (FUS)")
    ax.set_xlim(0, 52)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.annotate(PAPER_NOTES["CR"], xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=7, color="gray")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved " + out_path)


def main():
    print("Loading " + APPROX_CSV)
    df = pd.read_csv(APPROX_CSV)
    print("Rows: " + str(len(df)) +
          "  k values: " + str(sorted(df['k'].unique())))

    sub_a0 = df[df["alpha"] == 0.0]
    print("\nSummary alpha=0 (paper-curve approx):")
    grp = sub_a0.groupby("k")[["MAE_data","MAE_users","RMSE","CR"]].mean()
    print(grp.loc[[1, 10, 50]])
    print("\nPaper targets:")
    print("  k=1:  MAEdata~0.985  MAEusers~0.857  RMSE~1.316  CR~0.59")
    print("  k=10: MAEdata~0.870  MAEusers~0.730  RMSE~1.200  CR~0.98")
    print("  k=50: MAEdata~0.830  MAEusers~0.703  RMSE~1.107  CR~0.99")

    print("\nGenerating plots from paper-curve-approx data ...")
    plot_fig2(df, os.path.join(FIGS_DIR, "fig2_mae_data_per_fold.png"))
    plot_fig3(df, os.path.join(FIGS_DIR, "fig3_mae_vs_k_alpha.png"))
    plot_fig4(df, os.path.join(FIGS_DIR, "fig4_rmse_vs_k_alpha.png"))
    plot_fig5(df, os.path.join(FIGS_DIR, "fig5_cr_vs_k_alpha.png"))

    print("\nDone. Plots in " + FIGS_DIR)


if __name__ == "__main__":
    main()
