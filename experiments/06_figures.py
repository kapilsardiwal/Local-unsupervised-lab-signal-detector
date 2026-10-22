"""Poster figures from results/ -> figures/*.png (300 dpi) and *.svg."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def umap_panel(ax, emb, cat, colors, flag, title):
    for label, col in colors.items():
        m = cat == label
        ax.scatter(emb[m, 0], emb[m, 1], s=14 if col != GRID else 8, c=col, linewidths=0.6,
                   edgecolors=SURFACE, label=label, zorder=3 if col != GRID else 2)
    ax.scatter(emb[flag, 0], emb[flag, 1], s=70, facecolors="none", edgecolors=INK, linewidths=1.1,
               label="flagged", zorder=4)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=4, fontsize=9, markerscale=1.4, handletextpad=0.3)


def fig_hcv():
    s = pd.read_csv(RES / "hcv_scores.csv", index_col=0)
    emb = np.load(RES / "hcv_umap2d.npy")
    summ = json.load(open(RES / "hcv_summary.json"))
    grp = s["group"].replace({"Hepatitis": "Hepatitis / fibrosis / cirrhosis", "Fibrosis": "Hepatitis / fibrosis / cirrhosis",
                              "Cirrhosis": "Hepatitis / fibrosis / cirrhosis", "suspect Blood Donor": "Suspect donor"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    umap_panel(a, emb, grp.to_numpy(), {"Blood Donor": GRID, "Hepatitis / fibrosis / cirrhosis": BLUE,
                                        "Suspect donor": ORANGE}, s["flag"].to_numpy(),
               "UMAP of 10 labs (labels revealed after the run)")
    order = ["Blood Donor", "suspect Blood Donor", "Hepatitis", "Fibrosis", "Cirrhosis"]
    d = summ["detected_by_group"]
    rate = [d[g]["sum"] / d[g]["size"] * 100 for g in order]
    names = ["Blood donor", "Suspect donor", "Hepatitis C", "Fibrosis", "Cirrhosis"]
    y = np.arange(len(order))[::-1]
    b.barh(y, rate, height=0.5, color=[MUTED] + [BLUE] * 4)
    for yi, r, g in zip(y, rate, order):
        b.text(r + 1.5, yi, f"{d[g]['sum']}/{d[g]['size']}", va="center", color=INK2, fontsize=10)
    b.set_yticks(y, names), b.set_xlim(0, 110), b.set_xlabel("% flagged at FDR 5%")
    b.grid(axis="y", visible=False)
    b.set_title(f"Flagged: {summ['n_flagged']}, PPV {summ['ppv_flagged']:.0%}, AUROC {summ['auroc']:.2f}", loc="left")
    save(fig, "fig_hcv")


def fig_hcv_umap():
    """UMAP alone, for the poster (the bar chart is drawn natively in PowerPoint)."""
    s = pd.read_csv(RES / "hcv_scores.csv", index_col=0)
    emb = np.load(RES / "hcv_umap2d.npy")
    grp = s["group"].replace({"Hepatitis": "Hepatitis C / fibrosis / cirrhosis", "Fibrosis": "Hepatitis C / fibrosis / cirrhosis",
                              "Cirrhosis": "Hepatitis C / fibrosis / cirrhosis", "suspect Blood Donor": "Suspect donor",
                              "Blood Donor": "Blood donor"})
    fig, ax = plt.subplots(figsize=(7, 5.6))
    umap_panel(ax, emb, grp.to_numpy(), {"Blood donor": GRID, "Hepatitis C / fibrosis / cirrhosis": BLUE,
                                         "Suspect donor": ORANGE}, s["flag"].to_numpy(), "")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2, fontsize=11, markerscale=1.6, handletextpad=0.3)
    save(fig, "fig_hcv_umap")


def fig_power():
    R = pd.read_csv(RES / "adlb_power.csv")
    pats = [("hys_law", "Hy's Law"), ("aki_relative", "Creatinine 1.5–2× baseline"),
            ("plt_decline", "Platelets −35–50%, in range"), ("anion_gap", "Na↑ Cl↓, all in range"),
            ("fabricated", "Copied visits (data quality)")]
    g = R.groupby(["k", "pattern", "method"]).rate.agg(["mean", "sem"]).reset_index()
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharey=True)
    for ax, (p, title) in zip(axes, pats):
        for m, col, lab in [("pipeline", BLUE, "labsignal (nominal FDR 5%)"), ("rule_clin_sig", ORANGE, "Rule: >3×ULN, <0.5×LLN or Hy's Law")]:
            x = g[(g.pattern == p) & (g.method == m)]
            ax.fill_between(x.k, (x["mean"] - x["sem"]) * 100, (x["mean"] + x["sem"]) * 100, color=col, alpha=0.1, lw=0)
            ax.plot(x.k, x["mean"] * 100, color=col, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=lab)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0]), ax.set_ylim(-3, 103), ax.set_xlabel("effect size k")
    axes[0].set_ylabel("% of planted subjects found")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=11)
    fl = R[(R.pattern == "_n_flagged")].groupby("method").rate.mean()
    fig.text(0.01, -0.06, f"Mean subjects flagged per study (of 254): labsignal {fl['pipeline']:.0f}, "
             f"clinical rule {fl['rule_clin_sig']:.0f}, any out-of-range value {fl['rule_any_oor']:.0f}.",
             color=INK2, fontsize=10)
    save(fig, "fig_adlb_power")


def fig_nhanes():
    summ = json.load(open(RES / "nhanes_summary.json"))
    rows = []
    for key, label in [("panel_enrichment", "panel"), ("flag_enrichment", "any")]:
        for k, v in summ[key].items():
            rows.append({"name": k, "src": label, **v})
    d = pd.DataFrame(rows)
    d = d[d.cases_in_group >= 3].copy()
    # Woolf CI with 0.5 continuity correction
    n_rest = summ["n"] - d.group_n
    a, b = d.cases_in_group + .5, d.group_n - d.cases_in_group + .5
    c, e = d.prev_rest * n_rest + .5, n_rest - d.prev_rest * n_rest + .5
    lor, se = np.log(a * e / (b * c)), np.sqrt(1 / a + 1 / b + 1 / c + 1 / e)
    d["or"], d["lo"], d["hi"] = np.exp(lor), np.exp(lor - 1.96 * se), np.exp(lor + 1.96 * se)
    pretty = {"kidney_failure": "kidney failure", "dialysis": "dialysis", "liver_condition": "liver condition",
              "cirrhosis": "cirrhosis", "viral_hepatitis": "viral hepatitis", "diabetes": "diabetes",
              "insulin": "on insulin", "heart_failure": "heart failure", "cancer": "cancer", "pregnant": "pregnant"}
    d["label"] = [(f"{n.split('->')[0]} flag → {pretty[n.split('->')[1]]}" if s == "panel" else f"any flag → {pretty[n]}")
                  for n, s in zip(d.name, d.src)]
    d = pd.concat([d[d.src == "any"].sort_values("or"), d[d.src == "panel"].sort_values("or")])
    n_any = (d.src == "any").sum()
    y = np.r_[np.arange(n_any), np.arange(n_any, len(d)) + 1.5]  # gap for the section header
    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(d) + 1.6))
    col = np.where(d.src == "panel", BLUE, MUTED)
    ax.hlines(y, d.lo, d.hi, color=col, lw=2)
    ax.scatter(d["or"], y, c=col, s=50, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    xt = 3000
    for yi, o, n, ng in zip(y, d["or"], d.cases_in_group, d.group_n):
        ax.text(xt, yi, f"OR {o:.0f}" if o >= 10 else f"OR {o:.1f}", va="center", fontsize=9, color=INK2)
        ax.text(xt * 4, yi, f"{n}/{ng}", va="center", fontsize=9, color=MUTED)
    ax.text(0.2, y[-1] + 0.9, "Flag's most anomalous organ panel → matching diagnosis", fontsize=10.5,
            fontweight="bold", color=INK)
    ax.text(0.2, n_any - 0.4, "Any flag → diagnosis", fontsize=10.5, fontweight="bold", color=INK)
    ax.axvline(1, color=INK2, lw=1)
    ax.set_xscale("log"), ax.set_xlim(0.2, 1000)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(y, d.label.str.replace("any flag → ", "").str.replace(" flag → ", " → "))
    ax.set_ylim(-0.8, y[-1] + 1.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("odds ratio vs unflagged adults (log scale, 95% CI)")
    ax.set_title(f"NHANES: {summ['n_flagged']} of {summ['n']:,} adults flagged. Diagnoses were never shown to the model.",
                 loc="left")
    save(fig, "fig_nhanes_enrichment")


def fig_scaling():
    sc = json.load(open(RES / "scaling.json"))
    d = pd.DataFrame(sc["runs"])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(d.n, d.wall_s / 60, color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5)
    last = d.iloc[-1]
    ax.annotate(f"{last.n / 1000:.0f}k subjects\n{last.records_equiv / 1e6:.1f}M lab values\n"
                f"{last.wall_s / 60:.0f} min, {last.peak_mb / 1024:.1f} GB",
                (last.n, last.wall_s / 60), xytext=(-110, -10), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xlabel("subjects"), ax.set_ylabel("end-to-end minutes")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_title("Laptop runtime", loc="left")
    fig.text(0.01, -0.04, sc.get("machine") or "", color=MUTED, fontsize=8)
    save(fig, "fig_scaling")


if __name__ == "__main__":
    for f in [fig_hcv, fig_hcv_umap, fig_power, fig_nhanes, fig_scaling]:
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, "-", e.filename)
