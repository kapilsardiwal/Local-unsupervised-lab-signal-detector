"""labsignal dashboard: streamlit run app.py  (runs on localhost, no external calls)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import labsignal as ls
from labsignal.features import feature_lab
from labsignal.io import guess_columns, manifest, read_table

ROOT = Path(__file__).parent
DEMOS = {
    "CDISC pilot ADLB (chemistry + hematology)": ["data/raw/adlbc.xpt", "data/raw/adlbh.xpt"],
    "NHANES 2017-2020 adults (long format)": ["data/processed/nhanes_long.csv.gz"],
}

st.set_page_config(page_title="labsignal", layout="wide")
st.title("labsignal: local anomaly detection for lab data")
st.caption("UMAP + HDBSCAN structure, panel-wise Isolation Forest / LOF / robust PCA, "
           "bootstrap-calibrated FDR. Everything runs on this machine; no data leaves it.")

with st.sidebar:
    src = st.radio("Data", ["Upload", *DEMOS])
    if src == "Upload":
        up = st.file_uploader("ADLB-like or long-format file", type=["xpt", "sas7bdat", "csv", "parquet", "gz"])
        if not up:
            st.stop()
        raw_bytes = up.getvalue()
        df = read_table(up, up.name)
    else:
        df = pd.concat([read_table(ROOT / p) for p in DEMOS[src] if (ROOT / p).exists()])
        raw_bytes = None
        if "SAFFL" in df:
            df = df[(df["SAFFL"] == "Y") & ~df["AVISIT"].isin(["", "End of Treatment"])]

    g = guess_columns(df)
    opts = [None, *df.columns]
    st.subheader("Columns")
    cm = ls.ColumnMap(
        subject=st.selectbox("Subject", df.columns, list(df.columns).index(g.subject)),
        param=st.selectbox("Lab / parameter", df.columns, list(df.columns).index(g.param)),
        value=st.selectbox("Value", df.columns, list(df.columns).index(g.value)),
        time=st.selectbox("Study day (optional)", opts, opts.index(g.time)),
        lo=st.selectbox("Lower limit (optional)", opts, opts.index(g.lo)),
        hi=st.selectbox("Upper limit (optional)", opts, opts.index(g.hi)),
        baseline_flag=st.selectbox("Baseline flag (optional)", opts, opts.index(g.baseline_flag)),
    )
    fdr = st.slider("False discovery rate", 0.01, 0.20, 0.05, 0.01)
    seed = st.number_input("Seed", value=42, step=1)
    go_btn = st.button("Run", type="primary")


@st.cache_data(show_spinner=False)
def analyse(df, cm_dict, fdr, seed):
    cm = ls.ColumnMap(**cm_dict)
    long = ls.standardize(df, cm)
    F = ls.subject_features(long)
    res = ls.run(F, ls.Config(seed=int(seed), fdr=fdr))
    expl = ls.explain(res)
    return long, res, expl


if go_btn:
    st.session_state["key"] = (src, json.dumps(vars(cm)), fdr, seed)
if "key" not in st.session_state:
    st.info("Pick columns and press Run.")
    st.stop()

with st.spinner("Fitting detectors and calibrating against bootstrap nulls..."):
    long, res, expl = analyse(df, vars(cm), fdr, seed)
s = res.scores.join(expl)
flagged = s[s["flag"]].sort_values("p")

c = st.columns(5)
c[0].metric("Subjects", f"{len(s):,}")
c[1].metric("Lab records", f"{len(long):,}")
c[2].metric("Features", res.X.shape[1])
c[3].metric(f"Flagged (FDR {fdr:.0%})", f"{len(flagged)} ({len(flagged) / len(s):.1%})")
c[4].metric("Run time", f"{sum(v for k, v in res.timings.items() if k != 'boot_trim'):.0f} s")

tab_map, tab_flags, tab_subj, tab_audit = st.tabs(["Map", "Flagged subjects", "Subject detail", "Audit"])

with tab_map:
    e = pd.DataFrame(res.embedding2d, columns=["UMAP 1", "UMAP 2"], index=s.index)
    e["colour"] = np.where(s["flag"], s["top_panel"], "not flagged")
    e["cluster"] = s["cluster"].astype(str)
    left, right = st.columns(2)
    order = ["not flagged"] + sorted(set(e["colour"]) - {"not flagged"})
    fig = px.scatter(e.sort_values("colour", key=lambda x: x != "not flagged"), x="UMAP 1", y="UMAP 2",
                     color="colour", category_orders={"colour": order}, hover_name=e.index,
                     title="Flags by most anomalous panel", color_discrete_map={"not flagged": "#c8c8c8"})
    fig.update_traces(marker_size=5)
    left.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(e, x="UMAP 1", y="UMAP 2", color="cluster", hover_name=e.index,
                     title=f"HDBSCAN: {int(s['cluster'].max() + 1)} clusters, {(s['cluster'] == -1).sum()} noise")
    fig.update_traces(marker_size=5)
    right.plotly_chart(fig, use_container_width=True)

with tab_flags:
    st.dataframe(flagged[["p", "q", "top_panel", "drivers", "labs", "cluster"]], use_container_width=True)
    if len(flagged):
        hm = -np.log10(res.group_p.loc[flagged.index].clip(lower=1e-12))
        st.plotly_chart(px.imshow(hm.T, aspect="auto", color_continuous_scale="Blues",
                                  labels={"color": "-log10 p"}, title="Evidence by panel (flagged subjects)"),
                        use_container_width=True)

with tab_subj:
    sid = st.selectbox("Subject", list(flagged.index) + [x for x in s.index if x not in flagged.index])
    st.write(f"**p = {s.at[sid, 'p']:.2e}**, q = {s.at[sid, 'q']:.3f}, top panel: {s.at[sid, 'top_panel']}")
    labs = list(dict.fromkeys(feature_lab(f) for f in res.groups[s.at[sid, "top_panel"]]))
    shown = st.multiselect("Labs", sorted(long["param"].unique()),
                           [l for l in str(s.at[sid, "labs"]).split(", ") if l in set(long["param"])] or labs[:3])
    d = long[long["subject"] == sid]
    if "time" in d and d.groupby("param").size().max() > 1:
        for lab in shown:
            x = d[d["param"] == lab].sort_values("time")
            pop = long[long["param"] == lab]
            fig = go.Figure()
            if "hi" in x and x["hi"].notna().any():
                fig.add_hrect(y0=x["lo"].median(), y1=x["hi"].median(), fillcolor="#2a9d8f", opacity=0.12, line_width=0)
            fig.add_scatter(x=x["time"], y=x["value"], mode="lines+markers", name=sid)
            fig.update_layout(title=f"{lab} (shaded: reference range; population median {pop['value'].median():.3g})",
                              height=260, margin=dict(t=40, b=20), xaxis_title="study day")
            st.plotly_chart(fig, use_container_width=True)
    else:
        z = res.X.loc[sid, [c for c in res.X.columns if feature_lab(c) in shown]]
        st.plotly_chart(px.bar(z, orientation="h", labels={"value": "robust z vs population"}),
                        use_container_width=True)

with tab_audit:
    man = manifest(res, cm, input_bytes=raw_bytes)
    st.json(man)
    st.download_button("Subject scores (CSV)", s.to_csv(), "subject_scores.csv")
    st.download_button("Run manifest (JSON)", json.dumps(man, indent=2, default=str), "run_manifest.json")
