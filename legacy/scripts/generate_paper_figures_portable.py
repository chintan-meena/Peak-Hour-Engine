#!/usr/bin/env python3
"""Portable version of generate_paper_figures.py.

Same figures as the original, but with no machine-specific paths: it locates
the project by the folder it's run from (or --project), reads the target month
from the output filenames (or --month), and writes to <project>/Paper_Figures/.
This is the standalone twin of the notebook's SECTION 27, for regenerating the
figures without re-running the whole pipeline. Runs unchanged on macOS/Windows.

    python generate_paper_figures_portable.py                 # auto-detect
    python generate_paper_figures_portable.py --month 2026-09
    python generate_paper_figures_portable.py --project /path/to/NRLDC_Project
"""
import argparse, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import numpy as np


def block_to_time(block):
    minutes = (int(block) - 1) * 15
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def detect_month(artifact_dir, cli_month):
    if cli_month:
        return cli_month
    hits = sorted(artifact_dir.glob("RTM_Forecast_v3_*.csv"))
    if not hits:
        sys.exit(f"No RTM_Forecast_v3_*.csv in {artifact_dir}; pass --month.")
    m = re.search(r"RTM_Forecast_v3_(\d{4}-\d{2})\.csv", hits[-1].name)
    return m.group(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=".", help="Project root (default: current dir).")
    ap.add_argument("--month", default=None, help="Target month YYYY-MM (default: auto).")
    args = ap.parse_args()

    ROOT = Path(args.project).resolve()
    DATA = ROOT / "monthly_peak_pipeline_outputs"
    OUT = ROOT / "Paper_Figures"
    OUT.mkdir(parents=True, exist_ok=True)
    MONTH = detect_month(DATA, args.month)
    PLOT_DAY = f"{MONTH}-01"
    print(f"project={ROOT}  month={MONTH}  out={OUT}")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.7, "grid.linewidth": 0.4, "lines.linewidth": 1.1,
        "figure.dpi": 300, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.35,
    })
    COL_NL, COL_RTM, COL_DAM = "#1f5fa6", "#d9740b", "#5b5b5b"
    COL_WIN, COL_ACC = "#e07a5f", "#3b7d4f"
    SINGLE_W = 3.45

    def savefig(fig, name):
        p = OUT / name
        fig.savefig(p, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        print("saved", p)

    def time_to_block(hhmm):
        h, m = map(int, hhmm.strip().split(":"))
        return (h * 60 + m) // 15 + 1

    def selected_peak_blocks(default=range(74, 86)):
        s = None
        try:
            s = str(pd.read_csv(DATA / f"Monthly_Peak_Hours_{MONTH}.csv")["Peak_Hours"].iloc[0])
        except Exception:
            s = None
        if not s or s.lower() == "nan":
            return set(default)
        blocks = set()
        for rng in s.split(","):
            rng = rng.strip()
            if "-" in rng:
                a, b = rng.split("-")
                blocks.update(range(time_to_block(a), time_to_block(b)))
        return blocks or set(default)

    # ---- fig1 ----
    def fig1_framework():
        fig, ax = plt.subplots(figsize=(SINGLE_W, 7.6))
        ax.set_xlim(0, 6.2); ax.set_ylim(2.95, 15.25); ax.axis("off")
        def box(x, y, w, h, text, fc="#eef3f8", ec="#1f5fa6", fs=6.7, weight="normal"):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.07",
                                        linewidth=0.9, edgecolor=ec, facecolor=fc))
            ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                    weight=weight, linespacing=1.3)
            return (x, y, w, h)
        def arrow(b1, b2):
            x1,y1,w1,h1=b1; x2,y2,w2,h2=b2
            ax.add_patch(FancyArrowPatch((x1+w1/2,y1),(x2+w2/2,y2+h2),arrowstyle="-|>",
                         mutation_scale=7, linewidth=0.9, color="#333333", shrinkA=1, shrinkB=1))
        b_net=box(0.15,14.15,5.9,0.85,"Net NR Demand (historical, 15-min)",fc="#eaf1fb")
        b_wx =box(0.15,12.95,5.9,0.95,"Weather Composite\n(T, RH — 40-station, importance-weighted)",fc="#eaf7ee")
        b_mkt=box(0.15,11.65,5.9,1.05,"Market Signals\n(DAM/RTM price; exchange buy/sell/cleared volumes)",fc="#fbeee3")
        b_nlm=box(0.15,9.75,5.9,1.15,"Net-Load Model\nGradient-boosted regression trees (LightGBM)",fc="#dbe8fb",weight="bold")
        b_rtm=box(0.15,7.95,5.9,1.45,"RTM Price Model\nClimatology + 5-quantile ensemble, spike/crash\nregimes, conformal band",fc="#fbe2cd",ec="#d9740b",weight="bold")
        b_score=box(0.15,6.15,5.9,1.15,"Composite Block Score  S(t)\nrobust z-score fusion;\nNL × RTM interaction dominant (Eq. 4)",fc="#eaf3ea",ec="#3b7d4f",weight="bold")
        b_gate=box(0.15,4.55,2.85,1.05,"Evening-Ramp Gate\n(post afternoon-\ntrough start)",fc="#f5f5f5",ec="#555555",fs=6.3)
        b_agg=box(3.2,4.55,2.85,1.05,"Monthly\nAggregation\n(recency-weighted)",fc="#f5f5f5",ec="#555555",fs=6.3)
        b_out=box(0.15,3.15,5.9,0.95,"Declared Peak Window\n(fixed duration; continuous or split)",fc="#fdecec",ec="#c0392b",weight="bold")
        for a,b in [(b_net,b_nlm),(b_wx,b_nlm),(b_wx,b_rtm),(b_mkt,b_rtm),(b_nlm,b_rtm),
                    (b_nlm,b_score),(b_rtm,b_score),(b_score,b_gate),(b_score,b_agg),
                    (b_gate,b_out),(b_agg,b_out)]:
            arrow(a,b)
        savefig(fig,"fig1_framework.png")

    def fig2_weather_weights():
        df=pd.read_csv(ROOT/"City_Weather_Feature_Importance.csv").sort_values("Total_Importance").tail(20)
        fig,ax=plt.subplots(figsize=(SINGLE_W,4.0))
        ax.barh(df["City"],df["Total_Importance"],color=COL_NL,height=0.62)
        ax.set_xlabel("Aggregated feature importance (weighting basis)"); ax.set_ylabel("Weather station")
        ax.set_title("Top 20 of 40 NR load-centre weather stations\nby demand feature-importance weight",fontsize=8)
        ax.grid(axis="x"); ax.grid(axis="y",visible=False); savefig(fig,"fig2_weather_weights.png")

    def fig3_rtm_band():
        df=pd.read_csv(DATA/f"RTM_Forecast_v3_{MONTH}.csv",parse_dates=["Datetime"])
        day=df[pd.to_datetime(df["Date"]).dt.date==pd.Timestamp(PLOT_DAY).date()].copy(); t=day["Datetime"]
        fig,ax1=plt.subplots(figsize=(SINGLE_W,2.7))
        ax1.fill_between(t,day["Forecast_RTM_P05"],day["Forecast_RTM_P95"],color=COL_RTM,alpha=0.22,label="P05\u2013P95 conformal band",linewidth=0)
        ax1.plot(t,day["Forecast_RTM_Price"],color=COL_RTM,label="Forecast RTM price (median)")
        ax1.plot(t,day["DAM_Price"],color=COL_DAM,linestyle="--",linewidth=0.9,label="DAM price")
        ax1.set_ylabel("Price (Rs/MWh)"); ax1.set_xlabel(f"Time ({PLOT_DAY})")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax2=ax1.twinx(); ax2.plot(t,day["Forecast_RTM_Spike_Prob"],color="#8e2f2f",linewidth=0.9,alpha=0.85,label="Spike-regime probability")
        ax2.set_ylabel("Spike probability",color="#8e2f2f"); ax2.set_ylim(0,1.05); ax2.tick_params(axis="y",colors="#8e2f2f"); ax2.grid(False)
        l1,la1=ax1.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
        ax1.legend(l1+l2,la1+la2,loc="upper left",fontsize=6,framealpha=0.85)
        ax1.set_title("RTM price forecast with conformal band\nand spike-regime probability",fontsize=8); savefig(fig,"fig3_rtm_band.png")

    def fig4_cv_metrics():
        df=pd.read_csv(DATA/f"RTM_v3_WalkForwardCV_{MONTH}.csv")
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(SINGLE_W,5.1)); fig.subplots_adjust(hspace=0.55)
        x,w=df["Fold"],0.35
        ax1.bar(x-w/2,df["MAE"],width=w,color=COL_NL,label="MAE"); ax1.bar(x+w/2,df["RMSE"],width=w,color=COL_RTM,label="RMSE")
        ax1.set_xlabel("Walk-forward fold"); ax1.set_ylabel("Error (Rs/MWh)"); ax1.set_xticks(x); ax1.set_title("(a) Fold-wise MAE / RMSE",fontsize=8); ax1.legend()
        qc=["Pinball_Q5","Pinball_Q25","Pinball_Q50","Pinball_Q75","Pinball_Q85"]; ql=["Q05","Q25","Q50","Q75","Q85"]
        pres=[(c,l) for c,l in zip(qc,ql) if c in df.columns]
        ax2.bar([l for _,l in pres],[df[c].mean() for c,_ in pres],yerr=[df[c].std() for c,_ in pres],color=COL_ACC,capsize=3)
        ax2.set_xlabel("Quantile"); ax2.set_ylabel("Mean pinball loss (Rs/MWh)")
        ax2.set_title("(b) Pinball loss by quantile\n(mean \u00b1 s.d. across folds)",fontsize=8); savefig(fig,"fig4_cv_metrics.png")

    def fig5_threshold_sweep():
        df=pd.read_csv(DATA/f"RTM_v3_CapHitThresholdSweep_Combined_{MONTH}.csv")
        bi=df["F1_mean"].idxmax(); bt=df.loc[bi,"Threshold"]
        fig,ax=plt.subplots(figsize=(SINGLE_W,2.7))
        ax.plot(df["Threshold"],df["Precision_cv"],color=COL_NL,marker="o",ms=2.5,label="Precision")
        ax.plot(df["Threshold"],df["Recall_cv"],color=COL_RTM,marker="s",ms=2.5,label="Recall")
        ax.plot(df["Threshold"],df["F1_mean"],color=COL_ACC,marker="^",ms=2.5,label="F1 (mean)")
        ax.axvline(bt,color="#333333",linestyle=":",linewidth=1.0)
        ax.annotate(f"selected\nthr.={bt:.2f}",xy=(bt,df.loc[bi,"F1_mean"]),xytext=(bt+0.07,0.55),fontsize=6.3,arrowprops=dict(arrowstyle="->",lw=0.7))
        ax.set_xlabel("Cap-hit decision threshold"); ax.set_ylabel("Score"); ax.set_ylim(0,1.05)
        ax.set_title("Cap-hit classifier: precision, recall\nand F1 vs. decision threshold",fontsize=8); ax.legend(loc="lower left"); savefig(fig,"fig5_threshold_sweep.png")

    def fig6_method_comparison():
        df=pd.read_csv(DATA/f"Peak_Method_Comparison_{MONTH}.csv").sort_values("Score")
        colors=[COL_ACC if m=="recency_weighted" else "#9fb4c7" for m in df["Method"]]
        fig,ax=plt.subplots(figsize=(SINGLE_W,2.9))
        bars=ax.barh(df["Method"].str.replace("_"," "),df["Score"],color=colors)
        for b,hrs in zip(bars,df["Peak_Hours"]):
            ax.text(b.get_width()+0.4,b.get_y()+b.get_height()/2,hrs,va="center",fontsize=6.2,color="#333333")
        ax.set_xlabel("Aggregated window score")
        ax.set_title(f"Declared window by aggregation method\n({MONTH}; adopted method in green)",fontsize=8); savefig(fig,"fig6_method_comparison.png")

    def fig7_monthly_overview():
        df=pd.read_csv(DATA/f"RTM_Forecast_v3_{MONTH}.csv",parse_dates=["Datetime"])
        sel=selected_peak_blocks(); df["in_window"]=df["Block"].isin(sel)
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(SINGLE_W,4.3),sharex=True)
        ax1.plot(df["Datetime"],df["Net_Load"],color=COL_NL,linewidth=0.9)
        ax2.plot(df["Datetime"],df["Forecast_RTM_Price"],color=COL_RTM,linewidth=0.8)
        for _,g in df[df["in_window"]].groupby(df["Datetime"].dt.date):
            ax1.axvspan(g["Datetime"].min(),g["Datetime"].max(),color=COL_WIN,alpha=0.25,linewidth=0)
            ax2.axvspan(g["Datetime"].min(),g["Datetime"].max(),color=COL_WIN,alpha=0.25,linewidth=0)
        lab=f"{block_to_time(min(sel))}\u2013{block_to_time(max(sel)+1)}" if sel else "n/a"
        ax1.set_ylabel("Net NR load (MW)")
        ax1.set_title(f"Forecast net load and RTM price \u2014 target month {MONTH}\n(shaded: declared peak window, {lab} daily)",fontsize=8)
        ax2.set_ylabel("RTM price (Rs/MWh)"); ax2.set_xlabel("Date")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d")); ax2.xaxis.set_major_locator(mdates.DayLocator(interval=4))
        fig.autofmt_xdate(rotation=0,ha="center"); savefig(fig,"fig7_monthly_overview.png")

    def fig8_day_diagnostic():
        df=pd.read_csv(DATA/f"RTM_Forecast_v3_{MONTH}.csv",parse_dates=["Datetime"])
        day=df[pd.to_datetime(df["Date"]).dt.date==pd.Timestamp(PLOT_DAY).date()].copy()
        day["interaction"]=day["Net_Load"]*day["Forecast_RTM_Price"]
        sel=selected_peak_blocks(); win=day[day["Block"].isin(sel)]
        t0,t1=win["Datetime"].min(),win["Datetime"].max(); p90=day["Net_Load"].quantile(0.90)
        fig,axes=plt.subplots(3,1,figsize=(SINGLE_W,5.4),sharex=True); ax1,ax2,ax3=axes
        ax1.plot(day["Datetime"],day["Net_Load"],color=COL_NL); ax1.axhline(p90,color=COL_NL,linestyle="--",linewidth=0.8,alpha=0.7)
        ax1.set_ylabel("Net load (MW)")
        ax1.set_title(f"Net load, RTM price and their interaction \u2014 {PLOT_DAY}\n(shaded: selected window)",fontsize=8)
        ax2.plot(day["Datetime"],day["Forecast_RTM_Price"],color=COL_RTM); ax2.set_ylabel("RTM price\n(Rs/MWh)")
        ax3.fill_between(day["Datetime"],day["interaction"],color="#8e6fb0",alpha=0.55,linewidth=0)
        ax3.set_ylabel("NL \u00d7 RTM\n(MW\u00b7Rs/MWh)"); ax3.set_xlabel("Time of day")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); ax3.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax3.set_xlim(day["Datetime"].min(),day["Datetime"].max())
        for ax in axes:
            if pd.notna(t0) and pd.notna(t1): ax.axvspan(t0,t1,color=COL_WIN,alpha=0.3,linewidth=0)
        savefig(fig,"fig8_day_diagnostic.png")

    figs=[("fig1",fig1_framework),("fig2",fig2_weather_weights),("fig3",fig3_rtm_band),
          ("fig4",fig4_cv_metrics),("fig5",fig5_threshold_sweep),("fig6",fig6_method_comparison),
          ("fig7",fig7_monthly_overview),("fig8",fig8_day_diagnostic)]
    for name,fn in figs:
        try: fn()
        except Exception as e: print(f"  ! {name} skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
