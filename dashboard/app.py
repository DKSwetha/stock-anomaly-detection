"""
dashboard/app.py
-----------------
Streamlit dashboard for the Stock Anomaly Detection system.
Connects to the FastAPI backend at localhost:8000.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Anomaly Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Font Awesome + Google Fonts + Custom CSS ───────────────────────────────────
st.markdown("""
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&display=swap"
      rel="stylesheet">

<style>
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0A0F;
    color: #F5F5F7;
    font-family: 'Inter', sans-serif;
  }
  [data-testid="stAppViewContainer"] > .main { background-color: #0A0A0F; }
  [data-testid="stHeader"] { background: transparent; }
  h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em; }

  /* ── Header ── */
  .site-header {
    display: flex; align-items: center; gap: 14px;
    padding: 0.25rem 0 1.5rem 0;
    border-bottom: 1px solid #1f1f2e; margin-bottom: 2rem;
  }
  .site-header .icon-wrap {
    width: 48px; height: 48px; border-radius: 12px;
    background: linear-gradient(135deg, #F472B6, #7c3aed);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; color: #ffffff; flex-shrink: 0;
  }
  .site-header h1 { margin: 0; font-size: 1.75rem; font-weight: 700; color: #F5F5F7; }
  .site-header p  { margin: 2px 0 0 0; font-size: 0.875rem; color: #9ca3af; }

  /* ── Controls row — no card wrapper needed ── */
  .controls-label {
    font-family: 'Space Grotesk', sans-serif; font-size: 0.7rem;
    font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
    color: #9ca3af; margin-bottom: 0.4rem;
  }

  [data-testid="stSelectbox"] > div > div {
    background: #111118 !important; border: 1px solid #2a2a3d !important;
    border-radius: 10px !important; color: #f0eaff !important;
    font-family: 'Space Grotesk', sans-serif !important;
  }

  div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #F472B6 0%, #7c3aed 100%);
    color: #ffffff; border: none; border-radius: 10px;
    padding: 0.65rem 2rem; font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; font-size: 0.9rem; letter-spacing: 0.04em;
    cursor: pointer; width: 100%; margin-top: 20px;
    transition: opacity 0.15s ease;
  }
  div[data-testid="stButton"] > button:hover { opacity: 0.85; }
  /* Align button column to flex-end so it sits at the same baseline as the selectbox */
  div[data-testid="column"]:last-child {
    display: flex; flex-direction: column; justify-content: flex-end;
  }

  /* ── Stat cards ── */
  .stats-row { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
  .stat-card {
    flex: 1; min-width: 140px; background: #111118;
    border: 1px solid #1f1f2e; border-radius: 14px; padding: 1.1rem 1.4rem;
  }
  .stat-card .stat-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #9ca3af;
    font-family: 'Space Grotesk', sans-serif; margin-bottom: 0.3rem;
  }
  .stat-card .stat-value {
    font-size: 1.6rem; font-weight: 700;
    font-family: 'Space Grotesk', sans-serif; color: #F472B6; line-height: 1;
  }
  .stat-card .stat-sub { font-size: 0.78rem; color: #6b7280; margin-top: 0.25rem; }
  .stat-card.anomaly .stat-value { color: #f87171; }
  .stat-card.anomaly-zero .stat-value { color: #4ade80; }

  /* ── Section labels ── */
  .section-label {
    font-family: 'Space Grotesk', sans-serif; font-size: 0.7rem;
    font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
    color: #9ca3af; margin: 2rem 0 0.6rem 0;
    display: flex; align-items: center; gap: 8px;
  }
  .section-label i { color: #A855F7; font-size: 0.75rem; }

  /* ── Explain boxes ── */
  .chart-explain {
    background: #111118; border-left: 3px solid #A855F7;
    border-radius: 0 10px 10px 0; padding: 1rem 1.25rem;
    margin-top: 0.75rem; margin-bottom: 1.5rem;
    font-size: 0.85rem; color: #b8a8cc; line-height: 1.6;
  }
  .chart-explain strong { color: #F5F5F7; font-family: 'Space Grotesk', sans-serif; }

  /* ── Anomaly table ── */
  .anomaly-table {
    width: 100%; border-collapse: collapse;
    font-family: 'Space Grotesk', sans-serif; font-size: 0.82rem; margin-top: 0.5rem;
  }
  .anomaly-table th {
    text-align: left; padding: 0.6rem 0.8rem; color: #9ca3af;
    font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
    border-bottom: 1px solid #1f1f2e; background: #0f0f16;
  }
  .anomaly-table td { padding: 0.6rem 0.8rem; border-bottom: 1px solid #14141e; color: #F5F5F7; vertical-align: top; }
  .anomaly-table tr:last-child td { border-bottom: none; }
  .anomaly-table tr:hover td { background: #111118; }
  .anomaly-table .driver-vol   { color: #A855F7; font-weight: 600; }
  .anomaly-table .driver-price { color: #F472B6; font-weight: 600; }
  .feat-item {
    display: inline-block; margin: 2px 4px 2px 0;
    background: #1a1a2e; border: 1px solid #2a2a3d;
    border-radius: 4px; padding: 2px 6px;
    font-size: 0.74rem; color: #c4b5d4; font-family: 'Space Grotesk', monospace;
  }

  /* ── Error box ── */
  .error-box {
    background: rgba(168,85,247,0.08); border: 1px solid rgba(168,85,247,0.3);
    border-radius: 12px; padding: 1.2rem 1.5rem; color: #c4b5d4;
    font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem;
  }

  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"

TICKER_MAP = {
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corporation",
    "GOOGL": "Alphabet (Google)",
    "AMZN":  "Amazon.com Inc.",
    "META":  "Meta Platforms Inc.",
    "JPM":   "JPMorgan Chase",
    "BAC":   "Bank of America",
    "GS":    "Goldman Sachs",
    "XOM":   "ExxonMobil",
    "CVX":   "Chevron Corporation",
    "WMT":   "Walmart Inc.",
    "TGT":   "Target Corporation",
    "COST":  "Costco Wholesale",
    "JNJ":   "Johnson & Johnson",
    "PFE":   "Pfizer Inc.",
    "UNH":   "UnitedHealth Group",
    "DIS":   "The Walt Disney Company",
    "NKE":   "Nike Inc.",
    "TSLA":  "Tesla Inc.",
    "NVDA":  "NVIDIA Corporation",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0A0A0F",
    plot_bgcolor="#0D0D14",
    font=dict(family="Space Grotesk, Inter, sans-serif", color="#9ca3af", size=11),
    xaxis=dict(gridcolor="#1a1a28", zerolinecolor="#1a1a28", showgrid=True,
               tickfont=dict(color="#9ca3af")),
    yaxis=dict(gridcolor="#1a1a28", zerolinecolor="#1a1a28", showgrid=True,
               tickfont=dict(color="#9ca3af")),
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-header">
  <div class="icon-wrap"><i class="fa-solid fa-chart-line"></i></div>
  <div>
    <h1>Stock Anomaly Detector</h1>
    <p>LSTM Autoencoder — reconstruction error analysis across price and volume</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Controls — inline, no card wrapper ────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="controls-label"><i class="fa-solid fa-building"></i> &nbsp;Company</div>',
                unsafe_allow_html=True)
    display_names    = [f"{name}  ({ticker})" for ticker, name in TICKER_MAP.items()]
    ticker_keys      = list(TICKER_MAP.keys())
    selected_display = st.selectbox("Company", display_names, label_visibility="collapsed")
    selected_ticker  = ticker_keys[display_names.index(selected_display)]

with col2:
    analyze = st.button("Analyze", use_container_width=True)

st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

# ── Analysis ───────────────────────────────────────────────────────────────────
if analyze:
    with st.spinner("Fetching data and running inference..."):
        try:
            resp = requests.get(
                f"{API_BASE}/predict/{selected_ticker}",
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.markdown("""
            <div class="error-box">
              <i class="fa-solid fa-triangle-exclamation"></i> &nbsp;
              Cannot reach the API. Make sure the FastAPI server is running:<br>
              <code style="color:#A855F7;font-size:0.85rem">uvicorn api.main:app --reload</code>
            </div>""", unsafe_allow_html=True)
            st.stop()
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
              <i class="fa-solid fa-triangle-exclamation"></i> &nbsp;{e}
            </div>""", unsafe_allow_html=True)
            st.stop()

    ticker      = data["ticker"]
    full_name   = TICKER_MAP.get(ticker, ticker)
    dates       = data["dates"]
    prices      = data["prices"]
    errors      = data["reconstruction_errors"]
    is_anomaly  = data["is_anomaly"]
    threshold   = data["threshold"]
    n_anomalies = data["n_anomalies"]
    anomaly_det = data["anomaly_details"]

    anomaly_dates = [d for d, f in zip(dates, is_anomaly) if f]
    anomaly_close = [prices["Close"][i] for i, f in enumerate(is_anomaly) if f]

    # Period return for the stat cards
    first_close = prices["Close"][0]
    last_close  = prices["Close"][-1]
    period_ret  = ((last_close - first_close) / first_close) * 100
    ret_sign    = "+" if period_ret >= 0 else ""
    ret_color   = "#4ade80" if period_ret >= 0 else "#f87171"

    # ── Stat cards ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="stats-row">
      <div class="stat-card">
        <div class="stat-label"><i class="fa-solid fa-building-columns"></i> &nbsp;Ticker</div>
        <div class="stat-value">{ticker}</div>
        <div class="stat-sub">{full_name}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label"><i class="fa-regular fa-calendar-days"></i> &nbsp;Windows</div>
        <div class="stat-value">{len(dates)}</div>
        <div class="stat-sub">30-day sequences analyzed</div>
      </div>
      <div class="stat-card {'anomaly-zero' if n_anomalies == 0 else 'anomaly'}">
        <div class="stat-label"><i class="fa-solid fa-circle-exclamation"></i> &nbsp;Anomalies</div>
        <div class="stat-value">{n_anomalies}</div>
        <div class="stat-sub">above z-score threshold</div>
      </div>
      <div class="stat-card">
        <div class="stat-label"><i class="fa-solid fa-crosshairs"></i> &nbsp;Threshold</div>
        <div class="stat-value">{threshold:.5f}</div>
        <div class="stat-sub">mean + 1.5 std of errors</div>
      </div>
      <div class="stat-card">
        <div class="stat-label"><i class="fa-solid fa-dollar-sign"></i> &nbsp;Latest close</div>
        <div class="stat-value">${prices["Close"][-1]:,.2f}</div>
        <div class="stat-sub">as of {dates[-1]}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label"><i class="fa-solid fa-arrow-trend-up"></i> &nbsp;Period return</div>
        <div class="stat-value" style="color:{ret_color}">{ret_sign}{period_ret:.1f}%</div>
        <div class="stat-sub">{dates[0]} → {dates[-1]}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chart 1: Price ─────────────────────────────────────────────────────────
    st.markdown("""<div class="section-label">
      <i class="fa-solid fa-chart-area"></i> Price chart with anomaly markers
    </div>""", unsafe_allow_html=True)

    fig_price = go.Figure()
    # High — warm amber/orange line (top of band)
    fig_price.add_trace(go.Scatter(
        x=dates, y=prices["High"], mode="lines", name="High",
        line=dict(color="#4ade80", width=1.2),
        hovertemplate="High: $%{y:.2f}<extra></extra>",
    ))
    # Low — cool cyan line (bottom of band), fills up to High
    fig_price.add_trace(go.Scatter(
        x=dates, y=prices["Low"], mode="lines", name="Low",
        line=dict(color="#f87171", width=1.2),
        fill="tonexty", fillcolor="rgba(168,85,247,0.09)",
        hovertemplate="Low: $%{y:.2f}<extra></extra>",
    ))
    # Close — main bright purple line
    fig_price.add_trace(go.Scatter(
        x=dates, y=prices["Close"], mode="lines", name="Close",
        line=dict(color="#F472B6", width=2),
        hovertemplate="<b>%{x}</b><br>Close: $%{y:.2f}<extra></extra>",
    ))
    if anomaly_dates:
        fig_price.add_trace(go.Scatter(
            x=anomaly_dates, y=anomaly_close, mode="markers", name="Anomaly",
            marker=dict(color="#f87171", size=10, symbol="circle"),
            hovertemplate="<b>Anomaly: %{x}</b><br>Close: $%{y:.2f}<extra></extra>",
        ))
    fig_price.update_layout(
        height=340, margin=dict(l=0, r=0, t=12, b=0),
        legend=dict(orientation="h", x=0, y=1.08,
                    font=dict(size=11, color="#9ca3af"), bgcolor="rgba(0,0,0,0)"),
        yaxis_title="Price (USD)", **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="chart-explain">
      <strong>What this shows:</strong> The closing price over the selected period. The band
      between High and Low shows the daily trading range.
      <strong style="color:#F472B6">Anomaly markers</strong> flag windows where the model's
      reconstruction error exceeded the threshold — isolated markers suggest short-lived
      shocks; consecutive markers suggest a sustained shift in market behavior.
    </div>""", unsafe_allow_html=True)

    # ── Chart 2: Volume ────────────────────────────────────────────────────────
    st.markdown("""<div class="section-label">
      <i class="fa-solid fa-bars"></i> Trading volume
    </div>""", unsafe_allow_html=True)

    vol_colors = ["#f87171" if f else "#3b2f6b" for f in is_anomaly]
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=dates, y=prices["Volume"], name="Volume",
        marker_color=vol_colors,
        hovertemplate="<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>",
    ))
    fig_vol.update_layout(
        height=220, margin=dict(l=0, r=0, t=12, b=0),
        yaxis_title="Shares traded", showlegend=False, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""<div class="chart-explain">
      <strong>What this shows:</strong> Daily share volume. Bars coinciding with
      anomalous windows are highlighted in red.
      Volume spikes are frequently the <strong style="color:#F472B6">primary driver</strong>
      of anomalies — unusually high trading activity often reflects an external event
      such as an earnings release, macro news, or institutional rebalancing that disrupts
      the normal price-volume relationship the model learned.
    </div>""", unsafe_allow_html=True)

    # ── Chart 3: Reconstruction error ──────────────────────────────────────────
    st.markdown("""<div class="section-label">
      <i class="fa-solid fa-wave-square"></i> Reconstruction error over time
    </div>""", unsafe_allow_html=True)

    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=dates, y=errors, mode="lines+markers", name="Reconstruction error",
        line=dict(color="#F472B6", width=1.6),
        marker=dict(
            color=["#f87171" if f else "#F472B6" for f in is_anomaly],
            size=[9 if f else 4 for f in is_anomaly],
        ),
        hovertemplate="<b>%{x}</b><br>Error: %{y:.6f}<extra></extra>",
    ))
    fig_err.add_hline(
        y=threshold, line_color="#4ade80", line_dash="dash", line_width=1.5,
        annotation_text=f"Threshold  {threshold:.5f}",
        annotation_position="top right",
        annotation_font=dict(color="#4ade80", size=10, family="Space Grotesk"),
    )
    fig_err.update_layout(
        height=240, margin=dict(l=0, r=0, t=12, b=0),
        yaxis_title="MSE", showlegend=False, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_err, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""<div class="chart-explain">
      <strong>What this shows:</strong> The mean squared error between the model's
      reconstructed sequence and the actual 30-day window. The dashed line is the
      <strong style="color:#F472B6">anomaly threshold</strong> (mean + 1.5 standard
      deviations), computed fresh per ticker. Windows above this line are
      <strong style="color:#F472B6">flagged as anomalies</strong>. A rising error trend
      — even below the threshold — can indicate the stock is entering unfamiliar
      territory for the model.
    </div>""", unsafe_allow_html=True)

    # ── Anomaly breakdown table ─────────────────────────────────────────────────
    if anomaly_det:
        st.markdown("""<div class="section-label">
          <i class="fa-solid fa-table-list"></i> Anomaly breakdown by feature
        </div>""", unsafe_allow_html=True)

        rows = ""
        for d, detail in anomaly_det.items():
            err     = detail["reconstruction_error"]
            top     = detail["top_feature"]
            feat    = detail["feature_errors"]
            drv_cls = "driver-vol" if top == "Volume" else "driver-price"
            top_icon = (
                '<i class="fa-solid fa-chart-bar"></i>'
                if top == "Volume"
                else '<i class="fa-solid fa-arrow-trend-up"></i>'
            )
            feat_str = " ".join(
                f'<span class="feat-item">{k[0]}: {v:.4f}</span>'
                for k, v in feat.items()
            )
            rows += f"""<tr>
              <td>{d}</td>
              <td>{err:.6f}</td>
              <td class="{drv_cls}">{top_icon} &nbsp;{top}</td>
              <td>{feat_str}</td>
            </tr>"""

        st.markdown(f"""
        <table class="anomaly-table">
          <thead><tr>
            <th><i class="fa-regular fa-calendar"></i> &nbsp;Date</th>
            <th><i class="fa-solid fa-gauge-high"></i> &nbsp;Error score</th>
            <th><i class="fa-solid fa-fire"></i> &nbsp;Primary driver</th>
            <th><i class="fa-solid fa-sliders"></i> &nbsp;Per-feature errors</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
        <div class="chart-explain" style="margin-top:0.75rem">
          <strong>What this shows:</strong> For each anomalous window the reconstruction
          error is split by feature. The <strong style="color:#F472B6">primary driver</strong>
          is the feature with the highest individual error. A
          <strong style="color:#F472B6">Volume-driven</strong> anomaly means price moved
          normally but trading activity was unusual. A
          <strong style="color:#F472B6">price-driven</strong> anomaly (High, Low, Close)
          indicates unusual price movement independent of volume.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""<div class="chart-explain">
          <strong>No anomalies detected</strong> in the selected window. The stock's
          price-volume patterns are within the model's learned range of normal behavior
          for this period.
        </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem;">
      <i class="fa-solid fa-magnifying-glass-chart"
         style="font-size:3rem; color:#2a1f4a; display:block; margin-bottom:1rem;"></i>
      <p style="font-family:'Space Grotesk',sans-serif; font-size:1rem; color:#6b7280;">
        Select a company and analysis window above, then click Analyze.
      </p>
    </div>""", unsafe_allow_html=True)
