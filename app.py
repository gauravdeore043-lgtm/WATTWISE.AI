"""WattWise AI — Streamlit demo.

    streamlit run app.py
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from wattwise.anomaly import (LABEL_GAP, LABEL_HIGH_NORMAL, LABEL_LOW, LABEL_NORMAL, LABEL_STANDBY, LABEL_WASTE,
                              WASTE_LABELS, circuit_cols)
from wattwise.config import CIRCUIT_LABELS, REPORTS_DIR, SAMPLE_DIR, EconomicSettings
from wattwise.io import VALUE_KINDS, detect_columns, load_any, load_generic
from wattwise.pipeline import analyze, load_models
from wattwise.savings import standby_whatif

st.set_page_config(page_title="WattWise AI", page_icon="⚡", layout="wide")

# ------------------------------------------------------------------ design tokens
INK = "#16233B"
MUTED = "#5B6B80"
GROUND = "#F3F6F8"
BAND = "#A9C6E8"
EXPECT = "#2F6DB5"
LABEL_COLORS = {
    LABEL_WASTE: "#D7263D",
    LABEL_STANDBY: "#7B4FC9",
    LABEL_LOW: "#1F8A84",
    LABEL_HIGH_NORMAL: "#E39B26",
    LABEL_NORMAL: "#B7C2CF",
    LABEL_GAP: "#E5E9EE",
}
CIRCUIT_COLORS = {"c_kitchen": "#E39B26", "c_laundry": "#1F8A84", "c_water_heater_ac": "#2F6DB5", "c_other": "#B7C2CF"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');
html, body, p, li, label, input, textarea, button, h1, h2, h3, h4, [data-testid="stMetricValue"], [data-testid="stMarkdownContainer"] {{
  font-family: 'Atkinson Hyperlegible', sans-serif; }}
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {{ margin-top: 1rem; }}
.stApp {{ background: {GROUND}; color: {INK}; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
.verdict {{ font-size: 2.05rem; line-height: 1.2; font-weight: 700; color: {INK}; max-width: 46ch; margin: 0.2rem 0 0.6rem; }}
.verdict-sub {{ color: {MUTED}; font-size: 1.02rem; max-width: 75ch; }}
.grade {{ display: inline-flex; flex-direction: column; align-items: center; justify-content: center; width: 112px; height: 112px;
         border-radius: 50%; border: 6px solid var(--g); color: {INK}; background: white; }}
.grade b {{ font-size: 2.6rem; line-height: 1; }}
.grade span {{ font-size: .8rem; color: {MUTED}; }}
.rec {{ background: white; border-left: 5px solid {LABEL_COLORS[LABEL_WASTE]}; padding: .8rem 1rem; margin-bottom: .6rem; border-radius: 4px; }}
.rec.standby {{ border-left-color: {LABEL_COLORS[LABEL_STANDBY]}; }}
.rec.shift {{ border-left-color: {EXPECT}; }}
.rec h4 {{ margin: 0 0 .25rem; font-size: 1.02rem; }}
.rec p {{ margin: 0; color: {MUTED}; }}
.rec .num {{ float: right; font-weight: 700; color: {INK}; font-variant-numeric: tabular-nums; }}
.legend-dot {{ display:inline-block; width:.75rem; height:.75rem; border-radius:50%; margin-right:.35rem; vertical-align:-1px; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
</style>
""", unsafe_allow_html=True)

SAMPLES = {
    "Paris household, Sep–Nov 2010 (with circuits)": ("sample_meter_readings.csv", "FR"),
    "Same household with planted waste (answer key)": ("sample_with_planted_waste.csv", "FR"),
    "Total kWh only (no circuits)": ("sample_total_only.csv", "FR"),
    "Raw Kaggle/UCI minute-level file (Oct–Nov 2010)": ("sample_uci_minute_format.txt", "FR"),
}
COUNTRIES = {"India": "IN", "France": "FR", "United States": "US", "United Kingdom": "GB", "Germany": "DE", "None": None}


@st.cache_resource
def get_models():
    return load_models()


@st.cache_data(show_spinner=False)
def read_upload(data: bytes, name: str, ts, value, kind, circuits):
    src = io.BytesIO(data)
    if ts is None:
        return load_any(src)
    return load_generic(src, timestamp=ts, value=value, value_kind=kind, circuits=list(circuits))


def fmt_money(x, cur):
    return f"{cur}{x:,.0f}"


def base_layout(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Atkinson Hyperlegible, sans-serif", color=INK, size=13), hovermode="x unified",
                      showlegend=legend, legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, font=dict(size=12)))
    fig.update_xaxes(showgrid=False, linecolor="#D5DCE4")
    fig.update_yaxes(gridcolor="#EDF1F5", zeroline=False)
    return fig


def band_chart(df, title=None, height=360, show_flags=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["p90"], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=df["p10"], fill="tonexty", fillcolor="rgba(169,198,232,0.55)", line=dict(width=0),
                             name="Expected range (80%)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df.index, y=df["p50"], line=dict(color=EXPECT, width=1.4, dash="dot"), name="Expected"))
    fig.add_trace(go.Scatter(x=df.index, y=df["kwh"], line=dict(color=INK, width=1.3), name="Meter"))
    if show_flags and "label" in df:
        for lab in (LABEL_HIGH_NORMAL, LABEL_STANDBY, LABEL_LOW, LABEL_WASTE):
            m = df["label"].eq(lab)
            if m.any():
                fig.add_trace(go.Scatter(x=df.index[m], y=df["kwh"][m], mode="markers", name=lab,
                                         marker=dict(color=LABEL_COLORS[lab], size=7 if lab == LABEL_WASTE else 5)))
    fig.update_yaxes(title="kWh per hour")
    if title:
        fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)))
    return base_layout(fig, height)


# ================================================================== sidebar
with st.sidebar:
    st.markdown("## ⚡ WattWise AI")
    st.caption("Find wasted electricity in your meter readings and see what's coming next.")
    source = st.radio("Meter data", list(SAMPLES) + ["Upload my own file"], index=0)
    upload, mapping = None, {}
    if source == "Upload my own file":
        upload = st.file_uploader("CSV with a timestamp and consumption column", type=["csv", "txt"])
        st.caption("Any interval (1 min – 1 day). kWh, Wh, kW or W. Optional extra columns named `circuit_…` "
                   "are treated as sub-meters. The raw Kaggle/UCI file works as-is.")
        if upload is not None:
            head = upload.getvalue()[:400].decode("utf-8", "replace")
            if "Global_active_power" not in head:
                sep = ";" if head.split("\n")[0].count(";") > head.split("\n")[0].count(",") else ","
                peek = pd.read_csv(io.BytesIO(upload.getvalue()), sep=sep, nrows=200)
                g = detect_columns(peek)
                with st.expander("Column mapping", expanded=g["value"] is None):
                    cols = list(peek.columns)
                    ts = st.multiselect("Timestamp column(s)", cols, default=g["timestamp"])
                    val = st.selectbox("Consumption column", cols, index=cols.index(g["value"]) if g["value"] in cols else 0)
                    kind = st.selectbox("Its unit", VALUE_KINDS, index=VALUE_KINDS.index(g["value_kind"] or "energy_kwh"),
                                        format_func=lambda k: {"energy_kwh": "kWh per reading", "energy_wh": "Wh per reading",
                                                               "power_kw": "average kW", "power_w": "average W"}[k])
                    circ = st.multiselect("Sub-meter / circuit columns (optional)", [c for c in cols if c not in ts + [val]],
                                          default=g["circuits"])
                    mapping = {"ts": tuple(ts) if len(ts) > 1 else (ts[0] if ts else None), "value": val, "kind": kind, "circuits": tuple(circ)}

    default_country = SAMPLES[source][1] if source in SAMPLES else "IN"
    st.markdown("### Your tariff")
    econ = EconomicSettings()
    c1, c2 = st.columns(2)
    econ.currency = c1.text_input("Currency", "₹")
    econ.flat_rate = c2.number_input("Price per kWh", 0.0, 1000.0, 8.0, 0.5)
    econ.use_tou = st.toggle("Time-of-use tariff", False)
    if econ.use_tou:
        c1, c2 = st.columns(2)
        econ.peak_rate = c1.number_input("Peak price", 0.0, 1000.0, 10.0, 0.5)
        econ.offpeak_rate = c2.number_input("Off-peak price", 0.0, 1000.0, 6.0, 0.5)
        ph = st.slider("Peak hours", 0, 24, (18, 22))
        econ.peak_hours = tuple(range(ph[0], ph[1]))
        econ.offpeak_hours = tuple(h for h in list(range(22, 24)) + list(range(0, 6)) if h not in econ.peak_hours)
    econ.co2_kg_per_kwh = st.number_input("Grid emissions (kg CO₂ per kWh)", 0.0, 2.0, 0.71, 0.01)

    with st.expander("Model settings"):
        cname = st.selectbox("Public holidays of", list(COUNTRIES),
                             index=list(COUNTRIES.values()).index(default_country))
        horizon = st.slider("Forecast horizon (days)", 1, 14, 7)
        det_mode = st.radio("Learn 'normal' from", ["auto", "own", "pretrained"], horizontal=True,
                            format_func={"auto": "Auto", "own": "This meter", "pretrained": "Training household"}.get,
                            help="Auto learns from your own data when there are at least 4 weeks of it.")
        retrain = st.checkbox("Re-train the forecaster on this meter (≥ 120 days)", False)
        flex = st.slider("Movable share of peak usage (no circuits)", 0.0, 0.5, 0.15, 0.05)
    run = st.button("Analyse", type="primary", width="stretch")


# ================================================================== load + analyse
models = get_models()
key = (source, upload.name if upload else None, upload.size if upload else None, json.dumps(mapping, default=str),
       econ.flat_rate, econ.use_tou, econ.peak_rate, econ.offpeak_rate, econ.peak_hours, econ.co2_kg_per_kwh, econ.currency,
       cname, horizon, det_mode, retrain, flex)

if run or "result" not in st.session_state or st.session_state.get("key") != key and source in SAMPLES:
    try:
        if source in SAMPLES:
            path = SAMPLE_DIR / SAMPLES[source][0]
            hourly, info = read_upload(path.read_bytes(), path.name, None, None, None, ())
        elif upload is None:
            st.info("Choose a CSV file in the sidebar, or pick one of the samples.")
            st.stop()
        else:
            hourly, info = read_upload(upload.getvalue(), upload.name, mapping.get("ts"), mapping.get("value"),
                                       mapping.get("kind"), mapping.get("circuits", ()))
        with st.spinner("Learning this home's normal pattern, scoring every hour and forecasting…"):
            res = analyze(hourly, info, models, econ, country=COUNTRIES[cname], detector_mode=det_mode,
                          retrain_forecaster=retrain, horizon_days=horizon, flex_share=flex)
        st.session_state.update(result=res, key=key, source=source)
    except Exception as e:  # show a direction, not a stack trace
        st.error(f"Could not analyse this file: {e}. Check the column mapping in the sidebar "
                 "(timestamp, consumption column and its unit).")
        st.stop()

res = st.session_state["result"]
S, cur, F = res.summary, econ.currency, res.frame
grade_colors = {"A": "#1F8A84", "B": "#5BA35B", "C": "#E39B26", "D": "#E0662A", "E": "#D7263D"}

# ================================================================== hero: the verdict
left, right = st.columns([5, 1], vertical_alignment="center")
with left:
    lo, hi = S["avoidable_kwh_low"] * S["annual_factor"], S["avoidable_kwh_high"] * S["annual_factor"]
    st.markdown(f"<div class='verdict'>About {lo:,.0f}–{hi:,.0f} kWh a year looks avoidable — "
                f"roughly {fmt_money(S['annual_saving'], cur)} on your bill.</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='verdict-sub'>Based on {S['days_analyzed']:.0f} days ({F.index[0]:%d %b %Y} – {F.index[-1]:%d %b %Y}). "
        f"WattWise found {S['events']} unusual-consumption event{'s' if S['events'] != 1 else ''} and "
        f"{S['standby_events']} period{'s' if S['standby_events'] != 1 else ''} of elevated standby, "
        f"and treated {int(F['label'].eq(LABEL_HIGH_NORMAL).sum())} other high hours as normal variation for this home. "
        f"That is {S['avoidable_share']:.1%} of the {S['total_kwh']:,.0f} kWh used, or {S['annual_co2_kg']:,.0f} kg CO₂ a year.</div>",
        unsafe_allow_html=True)
with right:
    g = S["efficiency_grade"]
    st.markdown(f"<div class='grade' style='--g:{grade_colors[g]}'><b>{g}</b><span>{S['efficiency_score']}/100</span></div>",
                unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Average day", f"{S['avg_daily_kwh']:.1f} kWh", help="Mean daily consumption over the analysed period")
m2.metric("Always-on load", f"{S['median_baseload_w']:.0f} W", f"{S['median_baseload_w'] - S['best_baseload_w']:+.0f} W vs your best nights",
          delta_color="inverse")
nxt = res.forecast
nxt_day = res.forecaster.daily_totals(nxt)
wk_lo, wk_mid, wk_hi = res.forecaster.period_total(nxt_day)
m3.metric(f"Next {len(nxt_day)} full days", f"{wk_mid:.0f} kWh", f"{wk_lo:.0f}–{wk_hi:.0f} kWh likely range", delta_color="off")
m4.metric("Projected 30-day bill", fmt_money(nxt_day['p50'].mean() * 30 * S["avg_rate"], cur),
          help="Forecast daily average × 30 × your average price")

for n in res.notes:
    st.caption("• " + n)

tabs = st.tabs(["Waste report", "Usage timeline", "Forecast", "Standby load", "How it works", "Data & downloads"])

# ================================================================== tab 1: waste report
with tabs[0]:
    st.subheader("What to fix first")
    if not res.recommendations:
        st.success("No avoidable waste stood out in this period. Consumption stayed within what's normal for this home.")
    for r in res.recommendations:
        cls = "standby" if r["title"].startswith("Always-on") else "shift" if r["title"].startswith("Shift") else ""
        n = r.get("occurrences")
        unit = ("night" if cls == "standby" else "time") + ("s" if n != 1 else "")
        occ = f" ({n} {unit})" if n else ""
        st.markdown(f"<div class='rec {cls}'><span class='num'>{fmt_money(r['annual_saving'], cur)} a year, {r['annual_kwh']:,.0f} kWh</span>"
                    f"<h4>{r['title']}{occ}</h4><p>{r['action']}</p></div>", unsafe_allow_html=True)

    ev = res.events
    if len(ev):
        st.subheader("Unusual events")
        show = ev.assign(when=ev["start"].dt.strftime("%a %d %b %H:%M"), duration=ev["hours"].astype(str) + " h",
                         excess=ev["excess_kwh_low"].round(1).astype(str) + "–" + ev["excess_kwh_high"].round(1).astype(str) + " kWh",
                         cost=ev["cost_high"].map(lambda v: fmt_money(v, cur)))
        st.dataframe(show[["when", "duration", "severity", "cause", "excess", "cost", "peak_z"]], hide_index=True, width="stretch",
                     column_config={"when": "Started", "duration": "Lasted", "severity": "Severity", "cause": "Likely cause",
                                    "excess": "Extra energy", "cost": "Cost",
                                    "peak_z": st.column_config.NumberColumn("Unusualness (z)", format="%.1f")})

        st.subheader("Look closer at one event")
        opts = {f"{r.start:%a %d %b %H:%M}: {r.cause} ({r.excess_kwh_high:.1f} kWh)": i for i, r in ev.iterrows()}
        default = int(ev["excess_kwh_high"].idxmax())
        pick = opts[st.selectbox("Event", list(opts), index=list(opts.values()).index(default))]
        e = ev.loc[pick]
        win = F.loc[e["start"] - pd.Timedelta(hours=30): e["end"] + pd.Timedelta(hours=30)]
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = band_chart(win, "Meter vs what this home normally uses at these hours", 340)
            fig.add_vrect(x0=e["start"], x1=e["end"] + pd.Timedelta(hours=1), fillcolor=LABEL_COLORS[LABEL_WASTE], opacity=0.07, line_width=0)
            st.plotly_chart(fig, width="stretch")
        with c2:
            attr = e.get("attribution")
            if isinstance(attr, dict) and attr:
                a = pd.Series(attr).sort_values()
                fig = go.Figure(go.Bar(x=a.values * 100, y=a.index, orientation="h",
                                       marker_color=[LABEL_COLORS[LABEL_WASTE] if v == a.max() else "#B7C2CF" for v in a.values],
                                       text=[f"{v:.0%}" for v in a.values], textposition="outside"))
                fig.update_xaxes(title="share of the extra energy (%)", range=[0, 115])
                fig.update_layout(title=dict(text="Which circuit caused it", x=0, font=dict(size=15)))
                st.plotly_chart(base_layout(fig, 340, legend=False), width="stretch")
            else:
                st.markdown(f"**Likely cause:** {e['cause']}")
            st.markdown(f"**What to do:** {e['advice']}")
            st.caption(f"{e['actual_kwh']:.1f} kWh used vs {e['expected_kwh']:.1f} kWh expected over {e['hours']} h.")
        if e["kind"] == "waste":
            rec = res.detector.ae.reconstruct_day(F, e["start"])
            if rec is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rec.index, y=rec["reconstructed"], name="Typical shape the autoencoder expects",
                                         line=dict(color=EXPECT, dash="dot")))
                fig.add_trace(go.Scatter(x=rec.index, y=rec["actual"], name="That day", line=dict(color=INK)))
                fig.update_xaxes(title="hour of day", dtick=3)
                fig.update_yaxes(title="kWh")
                fig.update_layout(title=dict(text=f"Day shape on {e['start']:%d %b}: actual vs autoencoder reconstruction", x=0, font=dict(size=15)))
                st.plotly_chart(base_layout(fig, 280), width="stretch")

    if st.session_state.get("source", "").startswith("Same household with planted"):
        key_df = pd.read_csv(SAMPLE_DIR / "planted_waste_answer_key.csv", parse_dates=["start", "end"])
        flagged = F["label"].isin(WASTE_LABELS)
        key_df["found"] = [bool(flagged.loc[r.start:r.end].any()) for r in key_df.itertuples()]
        st.subheader("Answer key: planted events")
        st.caption(f"{key_df['found'].sum()} of {len(key_df)} planted waste events were detected.")
        st.dataframe(key_df[["start", "hours", "kw_added", "kwh_added", "description", "found"]], hide_index=True, width="stretch",
                     column_config={"found": st.column_config.CheckboxColumn("Detected")})

# ================================================================== tab 2: timeline
with tabs[1]:
    st.markdown(
        "Every hour is compared with what this home normally uses **at that hour, on that kind of day**. "
        "A busy evening that's typical for you is <span class='legend-dot' style='background:%s'></span>**high but normal**; "
        "the same usage at 3 am is <span class='legend-dot' style='background:%s'></span>**unusual**."
        % (LABEL_COLORS[LABEL_HIGH_NORMAL], LABEL_COLORS[LABEL_WASTE]), unsafe_allow_html=True)
    d0, d1 = F.index[0].date(), F.index[-1].date()
    rng = st.slider("Period", d0, d1, (max(d0, (F.index[-1] - pd.Timedelta(days=14)).date()), d1), format="DD MMM YYYY")
    sub = F.loc[str(rng[0]):str(rng[1]) + " 23:00"]
    st.plotly_chart(band_chart(sub, height=400), width="stretch")

    c1, c2 = st.columns([3, 2])
    with c1:
        piv = F.assign(date=F.index.date, hour=F.index.hour)
        z = piv.pivot_table(index="hour", columns="date", values="kwh", aggfunc="first")
        flag = piv.pivot_table(index="hour", columns="date", values="label", aggfunc="first")
        fig = go.Figure(go.Heatmap(z=z.values, x=[str(c) for c in z.columns], y=z.index, colorscale=[[0, "#F3F6F8"], [0.5, BAND], [1, INK]],
                                   colorbar=dict(title="kWh"), hovertemplate="%{x} %{y}:00<br>%{z:.2f} kWh<extra></extra>"))
        for lab in (LABEL_STANDBY, LABEL_WASTE):
            ys, xs = np.where((flag == lab).values)
            fig.add_trace(go.Scatter(x=[str(flag.columns[i]) for i in xs], y=flag.index[ys], mode="markers",
                                     marker=dict(symbol="square-open", size=7, color=LABEL_COLORS[lab], line=dict(width=1.5)),
                                     name=lab, hoverinfo="skip"))
        fig.update_yaxes(title="hour of day", autorange="reversed", dtick=3)
        fig.update_layout(title=dict(text="Hour × day map: red = unusual, purple = elevated standby", x=0, font=dict(size=15)))
        st.plotly_chart(base_layout(fig, 380, legend=False), width="stretch")
    with c2:
        counts = F["label"].value_counts().reindex([LABEL_WASTE, LABEL_STANDBY, LABEL_LOW, LABEL_HIGH_NORMAL, LABEL_NORMAL, LABEL_GAP]).fillna(0)
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=[LABEL_COLORS[l] for l in counts.index],
                               text=[f"{int(v):,} h" for v in counts.values], textposition="outside"))
        fig.update_xaxes(title="hours", range=[0, counts.max() * 1.25])
        fig.update_layout(title=dict(text="How every hour was classified", x=0, font=dict(size=15)))
        st.plotly_chart(base_layout(fig, 380, legend=False), width="stretch")

    if circuit_cols(F):
        cc = [c for c in ["c_water_heater_ac", "c_kitchen", "c_laundry", "c_other"] if c in F]
        prof = F.groupby(F.index.hour)[cc].mean()
        fig = go.Figure()
        for c in cc:
            fig.add_trace(go.Scatter(x=prof.index, y=prof[c], stackgroup="one", name=CIRCUIT_LABELS.get(c[2:], c[2:]),
                                     line=dict(width=0.5, color=CIRCUIT_COLORS.get(c, "#888"))))
        fig.update_xaxes(title="hour of day", dtick=3)
        fig.update_yaxes(title="average kWh")
        fig.update_layout(title=dict(text="Average day by circuit", x=0, font=dict(size=15)))
        st.plotly_chart(base_layout(fig, 320), width="stretch")

# ================================================================== tab 3: forecast
with tabs[2]:
    hist = F.iloc[-24 * 7:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["kwh"], line=dict(color=INK, width=1.2), name="Last 7 days (meter)"))
    fig.add_trace(go.Scatter(x=nxt.index, y=nxt["p90"], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=nxt.index, y=nxt["p10"], fill="tonexty", fillcolor="rgba(169,198,232,0.55)", line=dict(width=0),
                             name="80% prediction interval"))
    fig.add_trace(go.Scatter(x=nxt.index, y=nxt["p50"], line=dict(color=EXPECT, width=1.8), name="Forecast"))
    fig.update_yaxes(title="kWh per hour")
    fig.update_layout(title=dict(text=f"Hourly demand forecast — next {horizon} days", x=0, font=dict(size=15)))
    st.plotly_chart(base_layout(fig, 400), width="stretch")

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure(go.Bar(x=nxt_day.index.strftime("%a %d %b"), y=nxt_day["p50"], marker_color=EXPECT,
                               error_y=dict(type="data", symmetric=False, array=nxt_day["p90"] - nxt_day["p50"],
                                            arrayminus=nxt_day["p50"] - nxt_day["p10"], color=MUTED)))
        fig.update_yaxes(title="kWh per day")
        fig.update_layout(title=dict(text="Daily totals with range", x=0, font=dict(size=15)))
        st.plotly_chart(base_layout(fig, 320, legend=False), width="stretch")
    with c2:
        out = nxt_day[["p10", "p50", "p90"]].assign(cost=nxt_day["p50"] * S["avg_rate"])
        out.index = out.index.strftime("%a %d %b")
        st.dataframe(out.round(1), width="stretch",
                     column_config={"p10": "Low (P10)", "p50": "Expected (kWh)", "p90": "High (P90)",
                                    "cost": st.column_config.NumberColumn(f"Cost ({cur})", format="%.0f")})
        st.caption("Hourly bands are conformal prediction intervals, calibrated so about 8 in 10 future hours fall inside. "
                   "Daily ranges are calibrated separately on past daily totals (hourly ranges don't simply add up). "
                   "Only full days are listed.")

# ================================================================== tab 4: standby
with tabs[3]:
    sb = res.standby.dropna(subset=["baseload_kw"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sb.index, y=sb["best_nights_kw"] * 1000, line=dict(color=LABEL_COLORS[LABEL_LOW], dash="dot"),
                             name="Your best nights (same season)"))
    fig.add_trace(go.Scatter(x=sb.index, y=sb["baseload_kw"] * 1000, line=dict(color=INK), name="Nightly always-on load",
                             fill="tonexty", fillcolor="rgba(123,79,201,0.12)"))
    sd = F[F["label"].eq(LABEL_STANDBY)].index.normalize().unique()
    if len(sd):
        fig.add_trace(go.Scatter(x=sd, y=sb.reindex(sd)["baseload_kw"] * 1000, mode="markers", name="Elevated standby alert",
                                 marker=dict(color=LABEL_COLORS[LABEL_STANDBY], size=9)))
    fig.update_yaxes(title="watts")
    fig.update_layout(title=dict(text="Always-on load: the purple gap is avoidable", x=0, font=dict(size=15)))
    st.plotly_chart(base_layout(fig, 360), width="stretch")
    st.markdown(f"On a typical night your home's consumption doesn't drop below about **{S['median_baseload_w']:.0f} W**, "
                f"while on its best nights it gets down to **{S['best_baseload_w']:.0f} W**. "
                f"Closing that gap is worth **{S['standby_kwh'] * S['annual_factor']:,.0f} kWh a year**.")
    st.markdown("#### What if you removed some standby load?")
    w = st.slider("Watts of always-on load removed", 0, 500, int(max(10, round(S["median_baseload_w"] - S["best_baseload_w"], -1))), 10)
    wi = standby_whatif(w, econ)
    c1, c2, c3 = st.columns(3)
    c1.metric("Energy saved per year", f"{wi['annual_kwh']:,.0f} kWh")
    c2.metric("Money saved per year", fmt_money(wi["annual_saving"], cur))
    c3.metric("CO₂ avoided per year", f"{wi['annual_co2_kg']:,.0f} kg")

# ================================================================== tab 5: how it works
with tabs[4]:
    mpath = REPORTS_DIR / "metrics.json"
    st.markdown("""
**1. Expect.** A LightGBM quantile model predicts every hour from the previous days (lags, weekly pattern, calendar,
holidays), scale-normalised so it transfers between homes. Conformal calibration turns its P10–P90 into an honest
80% interval, re-calibrated on each uploaded meter's own recent errors.

**2. Compare in context.** Residuals are standardised separately for every hour-of-day × weekday/weekend, at 1, 3 and
6-hour windows. This is what separates *unusual* consumption from *normal daily variation*.

**3. Confirm.** An Isolation Forest (odd combinations of level, timing and circuit mix) and a neural autoencoder of
24-hour load shapes must agree with the residual evidence, unless the 6-hour evidence is strong on its own. A separate
change detector watches the nightly always-on floor for standby creep.

**4. Quantify.** Flagged hours become events with extra kWh (a range from conservative to expected), cost, CO₂ and,
where sub-meters exist, the circuit responsible. Standby is measured against the home's own best nights.
""")
    if mpath.exists():
        m = json.loads(mpath.read_text())
        st.markdown("#### Measured on held-out data (Jul–Nov 2010, never used for training or tuning)")
        c1, c2 = st.columns(2)
        with c1:
            t = pd.DataFrame(m["forecast"]["day_ahead_test"]).T[["MAE", "RMSE", "WAPE_%"]]
            st.markdown("**Day-ahead hourly forecast**")
            st.dataframe(t.round(3), width="stretch")
        with c2:
            t = pd.DataFrame(m["anomaly"]["test"]).T[["f1", "event_recall", "false_alarm_events_per_week"]]
            st.markdown("**Finding planted waste events** (5 random seeds)")
            st.dataframe(t.sort_values("f1", ascending=False).round(3), width="stretch",
                         column_config={"f1": "F1 (hours)", "event_recall": "Events found", "precision": "Precision",
                                        "false_alarm_events_per_week": "False alarms / week"})
        for img in ("detector_comparison.png", "forecast_horizon.png", "detection_example.png"):
            p = REPORTS_DIR / img
            if p.exists():
                st.image(str(p), width="stretch")

# ================================================================== tab 6: data
with tabs[5]:
    i = res.info
    st.markdown(f"**{i.source_format}**: {i.rows_raw:,} readings every {i.interval_minutes:g} min, turned into {i.hours:,} hours "
                f"({i.missing_hours} h of gaps, {i.imputed_hours} h interpolated).")
    if i.circuit_columns:
        st.markdown("**Circuits:** " + ", ".join(CIRCUIT_LABELS.get(c[2:], c[2:]) for c in i.circuit_columns))
    cols = ["kwh", "p10", "p50", "p90", "label", "z", "p_if", "p_ae", "anomaly_score"] + [c for c in F if c.startswith("c_")]
    out = F[[c for c in cols if c in F]].round(4)
    st.dataframe(out.tail(500), width="stretch", height=320)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download hourly results (CSV)", out.to_csv().encode(), "wattwise_hourly.csv", "text/csv", width="stretch")
    if len(res.events):
        c2.download_button("Download events (CSV)", res.events.drop(columns=["attribution"], errors="ignore").to_csv(index=False).encode(),
                           "wattwise_events.csv", "text/csv", width="stretch")
    c3.download_button("Download forecast (CSV)", nxt.round(4).to_csv().encode(), "wattwise_forecast.csv", "text/csv", width="stretch")
