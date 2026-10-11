"""SeamWise Edge: an inspectable, reproducible traffic-level simulation dashboard.

Run from this directory with: python -m streamlit run app.py
"""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from simulator import (
    Config,
    MODE_NAMES,
    PATHS,
    POLICIES,
    SCENARIOS,
    result_zip,
    run_comparison,
)


st.set_page_config(
    page_title="SeamWise Edge | Simulation Lab",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PATH_COLORS = {"Wi-Fi": "#35d6b2", "5G": "#75a9ff", "Satellite": "#ffcc75"}
POLICY_COLORS = {
    "reactive": "#aab6c7",
    "predictive": "#ba9cff",
    "protected": "#ffcc75",
    "adaptive_reactive": "#75a9ff",
    "seamwise": "#35d6b2",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stSidebar"] {border-right: 1px solid #263246;}
      [data-testid="stMetric"] {background: #142033; border: 1px solid #26364d;
        border-radius: 14px; padding: 16px; height: 100%;}
      [data-testid="stMetricLabel"] {color: #afbdd0;}
      h1 {letter-spacing: -.045em;}
      .eyebrow {color: #35d6b2; font-size: .78rem; font-weight: 700;
        letter-spacing: .16em; text-transform: uppercase; margin-bottom: .4rem;}
      .hero-copy {font-size: 1.07rem; color: #aebed2; max-width: 960px;}
      .run-tags {display: flex; gap: 8px; flex-wrap: wrap; margin: 14px 0 20px;}
      .run-tag {border: 1px solid #33465e; border-radius: 30px;
        padding: 5px 11px; font-size: .79rem; color: #c4d2e3;}
      .state-card {border: 1px solid #30415a; border-radius: 14px; padding: 17px 20px;
        background: #142033; margin-bottom: 14px; min-height: 210px;}
      .state-title {font-size: .84rem; color: #b3c3d8; margin-bottom: 10px;}
      .state-path {font-size: 1.55rem; font-weight: 700; margin: 3px 0;}
      .state-meta {color: #c6d1df; margin-bottom: 10px; font-size: .93rem;}
      .state-note {color: #9fadc1; font-size: .84rem; line-height: 1.55;}
      .good {color: #35d6b2;} .pending {color: #ffcc75;}
      .scope-note {border-left: 3px solid #35d6b2; padding: 8px 15px;
        color: #b6c6db; background: #132033; border-radius: 0 8px 8px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=16)
def calculate(seed: int, scenario: str, horizon: float, cap: float, all_policies: bool):
    """Cache the comparison for the selected experiment settings."""
    config = Config(
        seed=seed,
        scenario=scenario,
        duration_s=100.0,
        dt_s=0.25,
        prediction_horizon_s=horizon,
        duplicate_cap=cap,
    )
    policies = list(POLICIES) if all_policies else ["adaptive_reactive", "seamwise"]
    return run_comparison(config, policies=policies)


def style_figure(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=20, t=38, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111d2f",
        font=dict(family="Arial, sans-serif", color="#c5d2e4", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#253146", zeroline=False)
    fig.update_yaxes(gridcolor="#253146", zeroline=False)
    return fig


def percent(value) -> str:
    return "N/A" if value is None else f"{float(value):.1f}%"


def csv_bytes(frame: pd.DataFrame) -> bytes:
    # UTF-8 BOM lets Excel display Wi-Fi and mode labels without manual encoding.
    return frame.to_csv(index=False).encode("utf-8-sig")


with st.sidebar:
    st.markdown("### Experiment controls")
    st.caption("Settings take effect when you press **Run comparison**.")
    with st.form("experiment_settings"):
        scenario = st.selectbox(
            "Scenario",
            list(SCENARIOS),
            format_func=lambda key: key.replace("_", " ").title(),
        )
        seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
        horizon = st.slider("Prediction horizon (seconds)", 0.5, 5.0, 2.0, 0.5)
        cap_pct = st.slider("Duplicate payload cap (%)", 0, 15, 15, 1)
        all_policies = st.checkbox("Include all five policies", value=False)
        st.caption("100 simulated seconds · 0.25 s controller step")
        submitted = st.form_submit_button("Run comparison", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**A fair starting point**")
    st.caption(
        "The default reactive comparator has the same media adaptation and "
        "loss-triggered critical-message protection. SeamWise adds prediction "
        "and proactive decisions. All policies share one seeded environment."
    )
    with st.expander("What are the five policies?"):
        for key, label in POLICIES.items():
            st.markdown(f"**{label}**")
            explanations = {
                "reactive": "Reactive path selection with fixed full-video demand.",
                "predictive": "Prediction-based path selection with fixed full-video demand.",
                "protected": "Prediction and critical protection with fixed full-video demand.",
                "adaptive_reactive": "Reactive path selection, media adaptation and reactive protection.",
                "seamwise": "Prediction, proactive switching, critical protection and media adaptation.",
            }
            st.caption(explanations.get(key, key))
    st.caption("All traffic and network measurements are synthetic. No radio equipment is needed.")

if submitted or "results" not in st.session_state:
    with st.spinner("Simulating all selected policies on the same seeded network trace…"):
        computed = calculate(int(seed), scenario, float(horizon), cap_pct / 100.0, all_policies)
    st.session_state.results = computed
    st.session_state.applied = {
        "seed": int(seed), "scenario": scenario, "horizon": float(horizon),
        "cap_pct": cap_pct, "all_policies": all_policies,
    }
    st.session_state.export_zip = result_zip(computed)

results = st.session_state.results
applied = st.session_state.applied
seamwise = results["seamwise"]
baseline = results["adaptive_reactive"]
summary = pd.DataFrame([result.summary for result in results.values()])
timeline_frames = {key: pd.DataFrame(result.timeline) for key, result in results.items()}
network_frame = pd.DataFrame(seamwise.network)
last_sample = float(timeline_frames["seamwise"]["time_s"].iloc[-1])

st.markdown('<div class="eyebrow">SeamWise Edge · Simulation Lab</div>', unsafe_allow_html=True)
st.title("SeamWise Edge — Network & Results")
st.markdown(
    '<p class="hero-copy">Compare policies that predict network deterioration, switch paths, '
    'protect critical status and adapt media demand across Wi-Fi, 5G and satellite.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="scope-note"><b>Traffic-level simulation.</b> This dashboard models '
    'packets, delivery deadlines and media demand. It does not carry or play real video/audio, '
    'measure real networks, or demonstrate a working video call.</div>',
    unsafe_allow_html=True,
)
tags = [
    f"Applied scenario: {applied['scenario'].replace('_', ' ')}",
    f"Seed {applied['seed']}",
    "100 simulated seconds",
    f"Forecast +{applied['horizon']:g} s",
    f"Duplicate cap {applied['cap_pct']}%",
]
st.markdown(
    '<div class="run-tags">' + "".join(f'<span class="run-tag">{escape(t)}</span>' for t in tags) + "</div>",
    unsafe_allow_html=True,
)
st.caption(SCENARIOS[applied["scenario"]])

metrics = st.columns(4)
for column, key, label in [
    (metrics[0], "critical_success_pct", "Critical status on time"),
    (metrics[1], "mission_continuity_pct", "Fresh-status continuity"),
]:
    value, reference = seamwise.summary[key], baseline.summary[key]
    column.metric(label, percent(value), f"{value-reference:+.1f} pp vs adaptive reactive")
metrics[2].metric(
    "Usable video", f"{seamwise.summary['video_usable_s']:.1f} s",
    f"{seamwise.summary['video_usable_s']-baseline.summary['video_usable_s']:+.1f} s vs adaptive reactive",
)
metrics[3].metric(
    "Duplicate payload overhead", percent(seamwise.summary["duplicate_overhead_pct"]),
    f"{seamwise.summary['duplicate_payload_bytes']:,} extra payload bytes", delta_color="off",
)
st.caption(
    "Cards show the completed SeamWise run. "
    "pp = percentage points. A negative change is displayed as measured; no policy is guaranteed to win."
)

inspection_tab, result_tab, evidence_tab = st.tabs([
    "Network & decisions", "Comparison results", "Delivery evidence"
])


def network_plot(time_s: float, full_trace: bool, show_truth: bool) -> go.Figure:
    selected = network_frame if full_trace else network_frame[network_frame["time_s"] <= time_s]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=list(PATHS), vertical_spacing=0.085)
    for row, path in enumerate(PATHS, start=1):
        frame = selected[selected["path"] == path]
        color = PATH_COLORS[path]
        fig.add_trace(go.Scatter(
            x=frame["time_s"], y=frame["observed_capacity_mbps"],
            name="Observed capacity", legendgroup="observation", showlegend=row == 1,
            mode="lines", line=dict(color=color, width=2),
            hovertemplate="Observed: %{y:.2f} Mbps<extra></extra>",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=frame["time_s"] + applied["horizon"], y=frame["predicted_capacity_mbps"],
            name="Forecast at target time", legendgroup="forecast", showlegend=row == 1,
            mode="lines", line=dict(color=color, width=1.7, dash="dot"), opacity=0.7,
            hovertemplate="Forecast: %{y:.2f} Mbps<extra></extra>",
        ), row=row, col=1)
        if show_truth:
            fig.add_trace(go.Scatter(
                x=frame["time_s"], y=frame["capacity_mbps"],
                name="Simulator truth", legendgroup="truth", showlegend=row == 1,
                mode="lines", line=dict(color="#e0e7f1", width=1, dash="dash"), opacity=0.55,
                hovertemplate="Truth: %{y:.2f} Mbps<extra></extra>",
            ), row=row, col=1)
        fig.add_vline(x=time_s, line_width=1, line_color="#edf4ff", opacity=0.6, row=row, col=1)
        fig.update_yaxes(title_text="Mbps", rangemode="tozero", row=row, col=1)
    fig.update_xaxes(title_text="Simulation time (seconds)", range=[0, 100 + applied["horizon"]], row=3, col=1)
    return style_figure(fig, height=570)


def decisions_plot(time_s: float, full_trace: bool, selected_policies: list[str]) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                        subplot_titles=["Selected primary path", "Requested communication mode"])
    path_index = {path: index for index, path in enumerate(PATHS)}
    for policy in selected_policies:
        frame = timeline_frames[policy]
        if not full_trace:
            frame = frame[frame["time_s"] <= time_s]
        color = POLICY_COLORS[policy]
        common = dict(mode="lines", name=results[policy].label, legendgroup=policy,
                      line=dict(color=color, width=2.4, shape="hv"))
        fig.add_trace(go.Scatter(
            x=frame["time_s"], y=frame["primary"].map(path_index).fillna(-1),
            customdata=frame["primary"].fillna("Offline"),
            hovertemplate="%{customdata}<extra>%{fullData.name}</extra>", **common,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=frame["time_s"], y=frame["mode"], customdata=frame["mode_name"], showlegend=False,
            hovertemplate="%{customdata}<extra>%{fullData.name}</extra>", **common,
        ), row=2, col=1)
    for row in (1, 2):
        fig.add_vline(x=time_s, line_color="#edf4ff", opacity=0.6, line_width=1, row=row, col=1)
    fig.update_yaxes(tickvals=[-1, 0, 1, 2], ticktext=["Offline", *PATHS], range=[-1.3, 2.3], row=1, col=1)
    fig.update_yaxes(tickvals=list(MODE_NAMES), ticktext=list(MODE_NAMES.values()), range=[-0.3, 4.3], row=2, col=1)
    fig.update_xaxes(title_text="Simulation time (seconds)", range=[0, 100], row=2, col=1)
    return style_figure(fig, height=420)


with inspection_tab:
    st.markdown("### Inspect network conditions and controller decisions")
    st.caption(
        "Select a time in the completed simulation to inspect paths, communication modes, "
        "receiver status and network measurements."
    )
    t = float(st.slider("Inspect simulation time (seconds)", 0.0, last_sample, 0.0,
                        step=0.25, format="%.2f s", key="inspection_time"))
    step_index = min(int(round(t / 0.25)), len(timeline_frames["seamwise"]) - 1)
    st.caption(f"Selected sample: {t:.2f} s · Simulated duration: {seamwise.config['duration_s']:.2f} s")

    cards = st.columns(2)
    for column, policy in zip(cards, ["adaptive_reactive", "seamwise"]):
        record = timeline_frames[policy].iloc[step_index]
        status = "Fresh status at receiver" if bool(record["mission_fresh"]) else "No fresh status at receiver"
        status_class = "good" if bool(record["mission_fresh"]) else "pending"
        age = record["status_age_s"]
        age_text = "No status received yet" if pd.isna(age) else f"Latest status age: {age:.2f} s"
        protection = "Critical protection window active" if bool(record["protecting"]) else "No active protection window"
        with column:
            st.markdown(
                f'<div class="state-card"><div class="state-title">{escape(results[policy].label)}</div>'
                f'<div class="state-path" style="color:{POLICY_COLORS[policy]}">{escape(str(record["primary"] or "Offline"))}</div>'
                f'<div class="state-meta">{escape(str(record["mode_name"]))}</div>'
                f'<div class="state-note"><span class="{status_class}">{status}</span><br>'
                f'{age_text} · {int(record["critical_delivered"])} on-time status messages accepted<br>'
                f'{protection}</div></div>', unsafe_allow_html=True,
            )
            st.caption(f"Decision: {record['reason']}")

    instant = network_frame[network_frame["time_s"] == t].copy()
    instant_columns = {
        "path": "Path",
        "observed_capacity_mbps": "Observed capacity (Mbps)",
        "predicted_capacity_mbps": f"Forecast at {t + applied['horizon']:.2f} s (Mbps)",
        "observed_loss_pct": "Observed loss (%)",
    }
    if "observed_latency_ms" in instant:
        instant_columns["observed_latency_ms"] = "Observed one-way delay (ms)"
    st.dataframe(instant[list(instant_columns)].rename(columns=instant_columns),
                 use_container_width=True, hide_index=True,
                 column_config={name: st.column_config.NumberColumn(format="%.2f")
                                for key, name in instant_columns.items() if key != "path"})
    st.caption(
        "Simulated observations at this instant. A protection window permits extra copies, "
        "but the payload budget can suppress them; actual copies appear in Delivery evidence."
    )

    view_controls = st.columns([1.3, 1.3, 3])
    with view_controls[0]:
        full_trace = st.checkbox("Show full simulation trace", value=True, key="inspection_full_trace")
    with view_controls[1]:
        show_truth = st.checkbox("Show simulator truth", value=False, key="inspection_show_truth")
    with view_controls[2]:
        selected_policies = st.multiselect(
            "Policies on decision charts", list(results),
            default=["adaptive_reactive", "seamwise"], format_func=lambda p: results[p].label,
            key="inspection_chart_policies",
        )
    st.plotly_chart(network_plot(t, full_trace, show_truth), use_container_width=True, key="network_inspection")
    st.caption(
        "Dotted forecasts are shifted to their target time: observation time + prediction horizon. "
        "Each forecast uses only measurements already observed when it was made. "
        "The full trace and optional truth are for viewer inspection; controllers cannot see future truth."
    )
    st.plotly_chart(decisions_plot(t, full_trace, selected_policies), use_container_width=True, key="decision_inspection")
    st.caption("Requested mode is a traffic budget, not proof of successful reception. Usable media and fresh status are measured separately.")
    st.markdown("**Recent SeamWise events at or before this instant**")
    events = pd.DataFrame(seamwise.events)
    if not events.empty and "time_s" in events:
        recent = events[events["time_s"] <= t].tail(8).iloc[::-1]
        if recent.empty:
            st.info("No recorded events yet. Select a later simulation time.")
        else:
            st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("This run did not record any decision events.")


with result_tab:
    st.markdown("### Same environment. Measured outcomes.")
    st.caption(
        "All policies use the same scenario, random seed, noisy observations and seeded path-loss field. "
        "Different packet choices and service-slot use can still produce different realized deliveries."
    )
    score_fig = go.Figure()
    score_metrics = {
        "critical_success_pct": "Critical on-time delivery",
        "mission_continuity_pct": "Fresh-status continuity",
    }
    for policy, result in results.items():
        score_fig.add_bar(name=result.label, x=list(score_metrics.values()),
                          y=[result.summary[key] for key in score_metrics],
                          marker_color=POLICY_COLORS[policy], texttemplate="%{y:.1f}%", textposition="auto")
    score_fig.update_layout(barmode="group")
    score_fig.update_yaxes(title_text="Percent", range=[0, 105])
    st.plotly_chart(style_figure(score_fig, 320), use_container_width=True, key="comparison_scores")
    columns = {
        "label": "Policy", "critical_generated": "Status generated",
        "critical_on_time": "On time", "critical_success_pct": "On time (%)",
        "mission_continuity_pct": "Fresh status (%)", "video_usable_s": "Usable video (s)",
        "audio_usable_s": "Usable audio (s)", "switches": "Switches",
        "duplicate_overhead_pct": "Duplicate overhead (%)", "wire_bytes": "Wire bytes",
    }
    shown = summary[list(columns)].rename(columns=columns)
    st.dataframe(shown, use_container_width=True, hide_index=True,
                 column_config={name: st.column_config.NumberColumn(format="%.2f")
                                for name in shown.columns if "(%)" in name or "(s)" in name})
    st.markdown(
        "**Critical on-time delivery:** first valid copy received within 2 seconds of generation, "
        "including messages generated during outages. **Fresh-status continuity:** fraction of "
        "receiver time with a delivered status no more than 2 seconds old, evaluated every 0.1 seconds."
    )
    st.caption(
        "An audio/video generation epoch is usable if at least 80% of its offered payload arrives "
        "within 0.5 seconds. These are retrospective epoch scores. Audio and video usable times can "
        "overlap: video modes also carry audio. "
        "Duplicate overhead = duplicate payload bytes / original payload bytes. "
        "Headers, probes and ACKs are counted separately in total wire bytes."
    )
    with st.expander("Every reported metric and raw value", expanded=False):
        full_metrics = summary.set_index("label").T
        # String rendering preserves integers, None, labels and exact raw values together.
        st.dataframe(full_metrics.map(str) if hasattr(full_metrics, "map") else full_metrics.applymap(str),
                     use_container_width=True)
        st.caption(
            "The availability-conditioned score is supplementary: its denominator excludes messages "
            "generated when no path could serialize and propagate a critical packet to the receiver "
            "within 2 seconds even without competing traffic. "
            "It is not substituted for the all-message score."
        )
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download full results ZIP", st.session_state.export_zip,
        file_name=f"seamwise_results_{applied['scenario']}_seed{applied['seed']}.zip",
        mime="application/zip", use_container_width=True,
    )
    downloads[1].download_button(
        "Download comparison CSV", csv_bytes(summary), file_name="comparison.csv", mime="text/csv",
        use_container_width=True,
    )
    st.caption("The ZIP includes per-policy timelines, network observations, messages, transmissions, events and run metadata.")

with evidence_tab:
    st.markdown("### Follow a message to the receiver")
    selected = st.selectbox("Policy to inspect", list(results), index=list(results).index("seamwise"),
                            format_func=lambda key: results[key].label, key="evidence_policy")
    inspected = results[selected]
    messages = pd.DataFrame(inspected.messages)
    evidence_metrics = st.columns(4)
    evidence_metrics[0].metric("Generated status messages", f"{inspected.summary['critical_generated']:,}")
    evidence_metrics[1].metric("Delivered on time", f"{inspected.summary['critical_on_time']:,}")
    evidence_metrics[2].metric("Extra timely copies deduplicated", f"{inspected.summary['deduplicated_copies']:,}")
    evidence_metrics[3].metric("Copies blocked by budget", f"{inspected.summary['suppressed_copies']:,}")
    filters = st.columns(2)
    only_copied = filters[0].checkbox("Only messages sent on multiple paths")
    only_missed = filters[1].checkbox("Only messages that missed the deadline")
    filtered = messages.copy()
    if not filtered.empty:
        if only_copied:
            filtered = filtered[filtered["copies_sent"] > 1]
        if only_missed:
            filtered = filtered[~filtered["on_time"].astype(bool)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.caption(
            f"Showing {len(filtered)} of {len(messages)} critical messages. Blank first_arrival_s means "
            "no on-time copy was accepted; a late-only arrival is visible in the transmission log. "
            "copies_received includes late copies. The receiver counts an on-time message once; "
            "deduplicated_copies counts extra timely copies."
        )
    else:
        st.info("No messages were generated in this run.")
    st.download_button("Download this policy's message CSV", csv_bytes(messages),
                       file_name=f"{selected}_messages.csv", mime="text/csv")
    with st.expander("Transmission-level evidence"):
        transmissions = pd.DataFrame(inspected.transmissions)
        if not messages.empty and not transmissions.empty:
            message_id = st.selectbox("Critical message ID", messages["message_id"].tolist())
            packet_rows = transmissions[
                (transmissions["stream"] == "critical") & (transmissions["sequence"] == message_id)
            ]
            if packet_rows.empty:
                st.info("No critical packet transmission is recorded for this message (it may have been generated offline).")
            else:
                st.dataframe(packet_rows, use_container_width=True, hide_index=True)
            st.caption("A transmission row is an attempted packet copy. Its outcome and arrival expose loss, service-slot drops and lateness.")
        st.download_button("Download all transmission records", csv_bytes(transmissions),
                           file_name=f"{selected}_transmissions.csv", mime="text/csv")
    with st.expander("Complete decision event log"):
        st.dataframe(pd.DataFrame(inspected.events), use_container_width=True, hide_index=True)

