# app.py ───────────────────────────────────────────────────────────────
# Diffractometer geometry demo · AgBH calibration · selectable sources
# All displayed q-values in nm⁻¹
# Streamlit + Plotly
# ---------------------------------------------------------------------
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import gaussian_filter

# ──────────────────────────────────────────────────────────────────────
# Physical constants & global scale factor
HC      = 12.398_419_30               # keV·Å  (Planck·c)
d_AgBH  = 58.380                      # Å      (AgBH layer spacing)
q0      = 2 * np.pi / d_AgBH          # Å⁻¹    fundamental ring
Q_TO_NM = 10.0                        # 1 Å⁻¹ = 10 nm⁻¹  (display only)

KNOWN_SOURCES = {
    "Cu Kα (1.5406 Å)": 1.5406,
    "Mo Kα (0.7093 Å)": 0.7093,
    "Ag Kα (0.5594 Å)": 0.5594,
}

# ──────────────────────────────────────────────────────────────────────
# Streamlit layout & sidebar
st.set_page_config(page_title="Diffractometer + AgBH demo", layout="wide")
st.title("📐 Diffractometer geometry with AgBH calibration rings (q in nm⁻¹)")

with st.sidebar:
    st.header("Beam & detector parameters")

    src_choice = st.selectbox(
        "X-ray source",
        list(KNOWN_SOURCES.keys()) + ["Custom energy (keV)"],
        index=0,
    )

    if src_choice == "Custom energy (keV)":
        energy = st.slider("Photon energy (keV)", 8.0, 30.0, 22.16, 0.02)
        lam = HC / energy
    else:
        lam = KNOWN_SOURCES[src_choice]
        energy = HC / lam

    sd = st.slider("Sample → detector (mm)", 20.0, 3000.0, 1500.0, 10.0)

    det_side = st.slider("Detector side length (mm)", 10.0, 400.0, 200.0, 5.0)
    det_pix  = st.slider("Detector pixels per side", 128, 8096, 1024, 128)

    y_off = st.slider("Vertical detector offset (mm)", -200.0, 200.0, 0.0, 1.0)

    log_int = st.checkbox("Log-scale intensity", True)

# ──────────────────────────────────────────────────────────────────────
# Geometry numbers (still Å⁻¹ internally)
half = det_side / 2
y_top, y_bot = half + y_off, -half + y_off

theta_top = np.arctan2(abs(y_top), sd)
theta_bot = np.arctan2(abs(y_bot), sd)
q_top, q_bot = (4*np.pi/lam)*np.sin(theta_top), (4*np.pi/lam)*np.sin(theta_bot)
qmin, qmax = min(q_top, q_bot), max(q_top, q_bot)

# Convert to nm⁻¹ for display
qmin_nm, qmax_nm, q0_nm = qmin*Q_TO_NM, qmax*Q_TO_NM, q0*Q_TO_NM

# ──────────────────────────────────────────────────────────────────────
# Synthetic AgBH rings
y_mm = np.linspace(-half + y_off, half + y_off, det_pix)
z_mm = np.linspace(-half,            half,      det_pix)
Y, Z = np.meshgrid(y_mm, z_mm, indexing="ij")
R = np.sqrt(Y**2 + Z**2)
theta_pix = np.arctan2(R, sd)
q_pix = (4*np.pi/lam) * np.sin(theta_pix)          # Å⁻¹

sigma = 0.002
n_max = int(qmax / q0) + 2
I = np.zeros_like(q_pix)
for n in range(1, n_max + 1):
    I += (1/n**2) * np.exp(-0.5*((q_pix - n*q0)/sigma)**2)

I *= 1e4
#I += np.random.poisson(10, I.shape)
#I = gaussian_filter(I, sigma=1)

# what the user sees
I_plot  = np.log10(I + 1) if log_int else I
cb_ttl  = "log₁₀(counts)" if log_int else "counts"

# ──────────────────────────────────────────────────────────────────────
# FIRST ROW  – side-view wedge + rings
col_wedge, col_pat = st.columns([1.7, 1.3])

with col_wedge:
    st.markdown("### Intercepted angle (side view)")
    fig_w = go.Figure()

    fig_w.add_trace(go.Scatter(
        x=[0, sd], y=[0, 0], mode="lines",
        line=dict(color="gray", dash="dash"), showlegend=False))

    for y_edge in (y_top, y_bot):
        fig_w.add_trace(go.Scatter(
            x=[0, sd], y=[0, y_edge], mode="lines",
            line=dict(color="crimson"), showlegend=False))

    fig_w.add_shape(
        type="rect", x0=sd-2, x1=sd+2, y0=y_bot, y1=y_top,
        line=dict(color="RoyalBlue"),
        fillcolor="LightSkyBlue", opacity=0.25)

    deg = np.degrees
    fig_w.add_annotation(x=sd*0.55, y=y_top*0.55,
                         text=f"2θₘₐₓ ≈ {deg(2*theta_top):.2f}°", showarrow=False)
    fig_w.add_annotation(x=sd*0.55, y=y_bot*0.55,
                         text=f"2θₘᵢₙ ≈ {deg(2*theta_bot):.2f}°", showarrow=False)

    fig_w.update_layout(
        xaxis_title="x (mm)", yaxis_title="y (mm)",
        xaxis_range=[-sd*0.05, sd*1.05],
        yaxis_range=[-max(abs(y_top), abs(y_bot))*1.25,
                     +max(abs(y_top), abs(y_bot))*1.25],
        xaxis=dict(scaleanchor="y", scaleratio=1),
        width=560, height=380, margin=dict(l=20,r=20,b=20,t=30))
    st.plotly_chart(fig_w, use_container_width=True)

with col_pat:
    st.markdown("### AgBH calibration rings")
    fig2d = px.imshow(
        I_plot, origin="lower",
        color_continuous_scale="Turbo", aspect="equal")
    fig2d.update_layout(
        xaxis_visible=False, yaxis_visible=False,
        coloraxis_colorbar_title=cb_ttl,
        width=420, height=420, margin=dict(l=20,r=20,b=20,t=30))
    st.plotly_chart(fig2d, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────
# SECOND ROW – detector plane
st.markdown("---")
st.markdown("### Detector plane (front view)")

fig_det = go.Figure()
fig_det.add_shape(
    type="rect",
    x0=-half, y0=-half, x1=+half, y1=+half,
    xref="x", yref="y",
    line=dict(color="RoyalBlue"),
    fillcolor="LightSkyBlue", opacity=0.20)
fig_det.add_trace(go.Scatter(
    x=[0], y=[0], mode="markers+text",
    marker=dict(size=9, color="red"),
    text=["beam ⊙"], textposition="bottom center"))
fig_det.add_trace(go.Scatter(
    x=[0], y=[y_off], mode="markers+text",
    marker=dict(size=9, color="black"),
    text=[f"det ctr (y={y_off:.1f})"], textposition="top center"))
fig_det.update_layout(
    xaxis_title="horizontal (mm)", yaxis_title="vertical (mm)",
    xaxis_scaleanchor="y",
    width=700, height=450, margin=dict(l=20,r=20,b=20,t=30))
st.plotly_chart(fig_det, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────
# Summary table & 1-D cut
st.markdown(
f"""
### Geometry & AgBH peaks  

| parameter | value |
|-----------|-------|
| λ | **{lam:.4f} Å** |
| Source energy | **{energy:.2f} keV** |
| Sample–detector | **{sd:.1f} mm** |
| Detector size | **{det_side:.1f} × {det_side:.1f} mm** |
| Vertical offset | **{y_off:.1f} mm** |
| q<sub>min</sub> | **{qmin_nm:.3f} nm⁻¹** |
| q<sub>max</sub> | **{qmax_nm:.3f} nm⁻¹** |
| 2θ<sub>min</sub> | **{deg(2*theta_bot):.2f} °** |
| 2θ<sub>max</sub> | **{deg(2*theta_top):.2f} °** |
| AgBH d-spacing | **{d_AgBH:.3f} Å** |
| q₀ = 2π/d | **{q0_nm:.4f} nm⁻¹** |
""", unsafe_allow_html=True)

centre_row = I[I.shape[0]//2]
q_line_nm  = q_pix[q_pix.shape[0]//2] * Q_TO_NM

fig1d = go.Figure(go.Scatter(x=q_line_nm, y=centre_row, mode="lines"))
fig1d.update_layout(
    title="Horizontal cut – AgBH orders",
    xaxis_title="q (nm⁻¹)",
    yaxis_title=cb_ttl,
    width=820, height=340)
st.plotly_chart(fig1d, use_container_width=True)

st.caption(
    "All q-values now shown in **nm⁻¹** (Å⁻¹ × 10).  Physics remains unchanged; "
    "only the displayed units have shifted."
)
