import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OceanEmbed | Subsurface Thermal Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTS
# ============================================================

STANDARD_DEPTHS = np.array([
    0, 5, 10, 20, 30, 50, 75, 100,
    125, 150, 200, 300, 500, 700, 1000
])

RNG_SEED = 42

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #06111f;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #071827;
}

.hero {
    padding: 25px 30px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #071827 0%,
        #0b2942 50%,
        #062d35 100%
    );
    border: 1px solid rgba(0, 229, 255, 0.25);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 16px;
    color: #a8c4d8;
}

.status-card {
    padding: 18px;
    border-radius: 14px;
    background: #0b1e30;
    border: 1px solid rgba(255,255,255,0.08);
    min-height: 120px;
}

.status-title {
    font-size: 13px;
    color: #86a8bd;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.status-value {
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.small-note {
    color: #8ba6b8;
    font-size: 12px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
}

.info-box {
    background: #0a1d2c;
    border-left: 4px solid #00d9ff;
    padding: 15px 20px;
    border-radius: 8px;
}

.warning-box {
    background: #291d0a;
    border-left: 4px solid #ffb000;
    padding: 15px 20px;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA GENERATION
# ============================================================

@st.cache_data
def generate_ocean_domain(
    lat_min=5.0,
    lat_max=25.0,
    lon_min=55.0,
    lon_max=95.0,
    step=0.5
):

    rng = np.random.default_rng(RNG_SEED)

    lats = np.arange(lat_min, lat_max + step, step)
    lons = np.arange(lon_min, lon_max + step, step)

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Synthetic SST
    sst = (
        28.5
        + 2.0 * np.sin(np.radians(lat_grid * 2))
        - 1.5 * np.cos(np.radians(lon_grid * 1.5))
    )

    # Synthetic SSH / SLA
    ssh = (
        0.15 * np.sin(np.radians(lat_grid * 3))
        + 0.10 * np.cos(np.radians(lon_grid * 2))
    )

    # Synthetic SSS
    sss = (
        34.0
        + 1.5 * np.cos(np.radians(lat_grid))
        + 0.5 * np.sin(np.radians(lon_grid))
    )

    # Synthetic wind
    wind_u = (
        5.0 * np.sin(np.radians(lat_grid))
        + rng.normal(0, 0.2, lat_grid.shape)
    )

    wind_v = (
        3.0 * np.cos(np.radians(lon_grid))
        + rng.normal(0, 0.2, lat_grid.shape)
    )

    wind_speed = np.sqrt(wind_u ** 2 + wind_v ** 2)

    return (
        lats,
        lons,
        lat_grid,
        lon_grid,
        sst,
        ssh,
        sss,
        wind_u,
        wind_v,
        wind_speed
    )


# ============================================================
# PHYSICS-INFORMED MODEL
# ============================================================

def calculate_thermocline_depth(
    sst_val,
    ssh_val,
    sss_val,
    wind_speed
):
    """
    Physics-inspired thermocline estimate.

    Higher SSH:
        deeper warm layer

    Higher wind:
        stronger mixing

    Higher salinity:
        slightly modifies stratification
    """

    depth = (
        70
        + ssh_val * 180
        + (sst_val - 27.0) * 4
        + (sss_val - 35.0) * 5
        - wind_speed * 1.5
    )

    return float(np.clip(depth, 35, 180))


def reconstruct_vertical_profile(
    sst_val,
    ssh_val,
    sss_val,
    wind_speed,
    depths
):

    thermocline_depth = calculate_thermocline_depth(
        sst_val,
        ssh_val,
        sss_val,
        wind_speed
    )

    # Deep ocean reference temperature
    deep_temp = 4.0

    # Stratification parameter
    stratification = 2.2

    # Physics-inspired reconstruction
    pred_profile = (
        deep_temp
        + (sst_val - deep_temp)
        /
        (
            1
            + (depths / thermocline_depth)
            ** stratification
        )
    )

    # Wind-induced vertical mixing
    mixing_effect = (
        np.exp(-depths / 100)
        * wind_speed
        * 0.035
    )

    pred_profile -= mixing_effect

    # Salinity-based density adjustment
    salinity_effect = (
        (sss_val - 35.0)
        * 0.04
        * (depths / 500)
    )

    pred_profile -= salinity_effect

    # Surface boundary condition
    pred_profile[0] = sst_val

    return pred_profile, thermocline_depth


# ============================================================
# SYNTHETIC ARGO OBSERVATION
# ============================================================

def generate_argo_profile(pred_profile, sst_val):

    """
    Synthetic ARGO-like observation.

    IMPORTANT:
    This is still simulated data for demonstration purposes.
    """

    rng = np.random.default_rng(RNG_SEED)

    depth_bias = (
        0.08
        * np.sin(STANDARD_DEPTHS / 90)
    )

    noise = rng.normal(
        0,
        0.18,
        len(STANDARD_DEPTHS)
    )

    argo = (
        pred_profile
        + depth_bias
        + noise
    )

    argo[0] = sst_val

    return argo


# ============================================================
# OCEAN HEAT CONTENT
# ============================================================

def calculate_ohc(profile, depths):

    """
    Simplified relative Ocean Heat Content index.

    Not an operational physical OHC product.
    """

    reference_temp = 26.0

    warm_excess = np.maximum(
        profile - reference_temp,
        0
    )

    ohc_index = np.trapezoid(
        warm_excess,
        depths
    )

    return float(ohc_index)


# ============================================================
# MODEL CONFIDENCE
# ============================================================

def calculate_confidence(
    rmse,
    correlation,
    thermocline_depth
):

    score = 100

    score -= rmse * 30

    score += max(
        0,
        (correlation - 0.8) * 50
    )

    if thermocline_depth < 40:
        score -= 10

    return float(
        np.clip(score, 0, 100)
    )


# ============================================================
# LOAD DOMAIN
# ============================================================

(
    lats,
    lons,
    lat_grid,
    lon_grid,
    sst,
    ssh,
    sss,
    wind_u,
    wind_v,
    wind_speed
) = generate_ocean_domain()


# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🌊 OceanEmbed
</div>

<div class="hero-subtitle">
Physics-Informed AI for Subsurface Ocean Temperature Reconstruction
</div>

<br>

<div class="small-note">
North Indian Ocean • Satellite Surface Observations • ARGO Validation • 3D Digital Twin
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌐 OceanEmbed")

st.sidebar.markdown(
    "### Observation Controls"
)

selected_var = st.sidebar.selectbox(
    "Surface Variable",
    [
        "Sea Surface Temperature (SST)",
        "Sea Surface Height (SSH/SLA)",
        "Sea Surface Salinity (SSS)",
        "Surface Wind Speed"
    ]
)

sel_lat = st.sidebar.slider(
    "Target Latitude (°N)",
    float(lats.min()),
    float(lats.max()),
    15.0,
    step=0.5
)

sel_lon = st.sidebar.slider(
    "Target Longitude (°E)",
    float(lons.min()),
    float(lons.max()),
    68.0,
    step=0.5
)

st.sidebar.divider()

st.sidebar.markdown("### Model Configuration")

show_argo = st.sidebar.checkbox(
    "Show synthetic ARGO",
    True
)

show_3d = st.sidebar.checkbox(
    "Enable 3D reconstruction",
    True
)

st.sidebar.divider()

st.sidebar.caption(
    "⚠️ Demonstration system using synthetic observations. "
    "Not an operational ocean forecast."
)


# ============================================================
# TARGET LOCATION
# ============================================================

lat_idx = np.abs(
    lats - sel_lat
).argmin()

lon_idx = np.abs(
    lons - sel_lon
).argmin()

curr_sst = float(
    sst[lat_idx, lon_idx]
)

curr_ssh = float(
    ssh[lat_idx, lon_idx]
)

curr_sss = float(
    sss[lat_idx, lon_idx]
)

curr_wind = float(
    wind_speed[lat_idx, lon_idx]
)


# ============================================================
# MODEL INFERENCE
# ============================================================

pred_temp, thermocline_depth = reconstruct_vertical_profile(
    curr_sst,
    curr_ssh,
    curr_sss,
    curr_wind,
    STANDARD_DEPTHS
)

argo_temp = generate_argo_profile(
    pred_temp,
    curr_sst
)

# ============================================================
# METRICS
# ============================================================

rmse = float(
    np.sqrt(
        np.mean(
            (pred_temp - argo_temp) ** 2
        )
    )
)

bias = float(
    np.mean(
        pred_temp - argo_temp
    )
)

corr = float(
    np.corrcoef(
        pred_temp,
        argo_temp
    )[0, 1]
)

ohc = calculate_ohc(
    pred_temp,
    STANDARD_DEPTHS
)

confidence = calculate_confidence(
    rmse,
    corr,
    thermocline_depth
)


# ============================================================
# KPI ROW
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Surface SST",
    f"{curr_sst:.2f} °C"
)

c2.metric(
    "Thermocline",
    f"{thermocline_depth:.1f} m"
)

c3.metric(
    "Profile RMSE",
    f"{rmse:.3f} °C"
)

c4.metric(
    "ARGO Correlation",
    f"{corr:.3f}"
)

c5.metric(
    "Model Confidence",
    f"{confidence:.0f}%"
)


st.divider()


# ============================================================
# TABS
# ============================================================

(
    tab_dashboard,
    tab_physics,
    tab_risk,
    tab_architecture
) = st.tabs(
    [
        "🚀 Mission Dashboard",
        "🧠 Physics + AI",
        "🌪️ Heat & Cyclone Risk",
        "🛠️ System Architecture"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with tab_dashboard:

    col_map, col_profile = st.columns(
        [1.25, 1]
    )

    # --------------------------------------------------------
    # SURFACE MAP
    # --------------------------------------------------------

    with col_map:

        st.subheader(
            f"🛰️ Surface Observation: {selected_var}"
        )

        if selected_var == "Sea Surface Temperature (SST)":

            data_to_plot = sst
            color_scale = "Turbo"
            unit = "°C"

        elif selected_var == "Sea Surface Height (SSH/SLA)":

            data_to_plot = ssh
            color_scale = "Balance"
            unit = "m"

        elif selected_var == "Sea Surface Salinity (SSS)":

            data_to_plot = sss
            color_scale = "Viridis"
            unit = "PSU"

        else:

            data_to_plot = wind_speed
            color_scale = "Inferno"
            unit = "m/s"

        fig_map = px.imshow(
            data_to_plot,
            x=lons,
            y=lats,
            origin="lower",
            color_continuous_scale=color_scale,
            labels={
                "x": "Longitude (°E)",
                "y": "Latitude (°N)",
                "color": unit
            },
            aspect="auto"
        )

        fig_map.add_trace(
            go.Scatter(
                x=[sel_lon],
                y=[sel_lat],
                mode="markers+text",
                marker=dict(
                    size=15,
                    color="red",
                    symbol="cross"
                ),
                text=[
                    f"PROBE<br>"
                    f"{sel_lat:.1f}°N, "
                    f"{sel_lon:.1f}°E"
                ],
                textposition="top center",
                name="Virtual Probe"
            )
        )

        fig_map.update_layout(
            height=500,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            )
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True
        )


    # --------------------------------------------------------
    # PROFILE
    # --------------------------------------------------------

    with col_profile:

        st.subheader(
            "🌡️ Reconstructed Vertical Profile"
        )

        fig_profile = go.Figure()

        fig_profile.add_trace(
            go.Scatter(
                x=pred_temp,
                y=STANDARD_DEPTHS,
                mode="lines+markers",
                name="OceanEmbed Reconstruction",
                line=dict(
                    color="#00d9ff",
                    width=4
                ),
                marker=dict(size=6)
            )
        )

        if show_argo:

            fig_profile.add_trace(
                go.Scatter(
                    x=argo_temp,
                    y=STANDARD_DEPTHS,
                    mode="markers",
                    name="Synthetic ARGO",
                    marker=dict(
                        color="#ff9f43",
                        size=8,
                        symbol="diamond"
                    )
                )
            )

        fig_profile.add_hline(
            y=thermocline_depth,
            line_dash="dash",
            line_color="#ff4d6d",
            annotation_text=(
                f"Thermocline ≈ "
                f"{thermocline_depth:.0f} m"
            )
        )

        fig_profile.update_yaxes(
            autorange="reversed",
            title="Depth (m)"
        )

        fig_profile.update_xaxes(
            title="Temperature (°C)"
        )

        fig_profile.update_layout(
            height=500,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),
            legend=dict(
                orientation="h",
                y=1.05
            )
        )

        st.plotly_chart(
            fig_profile,
            use_container_width=True
        )


    # ========================================================
    # OBSERVATION SUMMARY
    # ========================================================

    st.subheader("📡 Current Ocean State")

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "SSH / SLA",
        f"{curr_ssh:.3f} m"
    )

    s2.metric(
        "SSS",
        f"{curr_sss:.2f} PSU"
    )

    s3.metric(
        "Wind Speed",
        f"{curr_wind:.2f} m/s"
    )

    s4.metric(
        "Ocean Heat Index",
        f"{ohc:.1f}"
    )


    # ========================================================
    # 3D VOLUME
    # ========================================================

    if show_3d:

        st.divider()

        st.subheader(
            "🌐 3D Subsurface Thermal Digital Twin"
        )

        sub_lons = np.linspace(
            sel_lon - 4,
            sel_lon + 4,
            15
        )

        sub_lats = np.linspace(
            sel_lat - 4,
            sel_lat + 4,
            15
        )

        X, Y, Z = np.meshgrid(
            sub_lons,
            sub_lats,
            STANDARD_DEPTHS,
            indexing="xy"
        )

        local_thermocline = (
            thermocline_depth
            * (
                1
                + 0.08
                * np.sin(
                    np.radians(X * 5)
                )
            )
        )

        T_3D = (
            4.0
            +
            (curr_sst - 4.0)
            /
            (
                1
                +
                (Z / local_thermocline) ** 2.2
            )
        )

        fig_3d = go.Figure(
            data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=T_3D.flatten(),
                isomin=4,
                isomax=31,
                opacity=0.12,
                surface_count=20,
                colorscale="Inferno",
                colorbar=dict(
                    title="Temperature °C"
                )
            )
        )

        fig_3d.update_layout(
            height=650,
            scene=dict(
                xaxis_title="Longitude °E",
                yaxis_title="Latitude °N",
                zaxis=dict(
                    title="Depth (m)",
                    autorange="reversed"
                ),
                aspectratio=dict(
                    x=1,
                    y=1,
                    z=0.8
                )
            ),
            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0
            )
        )

        st.plotly_chart(
            fig_3d,
            use_container_width=True
        )


    # ========================================================
    # DOWNLOAD PROFILE
    # ========================================================

    profile_df = pd.DataFrame({
        "Depth_m": STANDARD_DEPTHS,
        "OceanEmbed_Temperature_C": pred_temp,
        "Synthetic_ARGO_Temperature_C": argo_temp
    })

    csv_data = profile_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Reconstructed Profile",
        csv_data,
        "oceanembed_profile.csv",
        "text/csv"
    )


# ============================================================
# PHYSICS TAB
# ============================================================

with tab_physics:

    st.header(
        "🧠 How Can Surface Data Reveal the Deep Ocean?"
    )

    st.markdown("""
    Satellites cannot directly observe the full temperature
    profile of the ocean. However, subsurface structure leaves
    measurable signatures at the surface.
    """)

    p1, p2, p3 = st.columns(3)

    with p1:

        st.markdown("""
        ### 🌊 SSH

        Warm water expands and increases sea level.

        Therefore:

        **Higher SSH → deeper warm layer**

        SSH becomes an important proxy for
        thermocline variability.
        """)

    with p2:

        st.markdown("""
        ### 🌡️ SST

        Sea Surface Temperature provides the
        boundary condition for the reconstructed
        thermal profile.

        The model uses SST to estimate how much
        warm water is available near the surface.
        """)

    with p3:

        st.markdown("""
        ### 🧂 SSS + 🌪️ Wind

        Salinity affects water density and
        stratification.

        Wind controls vertical mixing and
        can modify the thickness of the
        surface mixed layer.
        """)

    st.divider()

    st.subheader(
        "Interactive Physics Sandbox"
    )

    test_ssh = st.slider(
        "Sea Surface Height Anomaly (m)",
        -0.5,
        0.5,
        0.1,
        0.05
    )

    test_sst = st.slider(
        "Sea Surface Temperature (°C)",
        20.0,
        32.0,
        29.0,
        0.5
    )

    test_sss = st.slider(
        "Sea Surface Salinity (PSU)",
        32.0,
        37.0,
        35.0,
        0.1
    )

    test_wind = st.slider(
        "Surface Wind Speed (m/s)",
        0.0,
        15.0,
        5.0,
        0.5
    )

    sim_pred, sim_thermocline = reconstruct_vertical_profile(
        test_sst,
        test_ssh,
        test_sss,
        test_wind,
        STANDARD_DEPTHS
    )

    fig_sim = go.Figure()

    fig_sim.add_trace(
        go.Scatter(
            x=sim_pred,
            y=STANDARD_DEPTHS,
            mode="lines+markers",
            line=dict(
                color="#00d9ff",
                width=4
            ),
            name="Predicted Profile"
        )
    )

    fig_sim.add_hline(
        y=sim_thermocline,
        line_dash="dash",
        annotation_text=(
            f"Thermocline: "
            f"{sim_thermocline:.1f} m"
        )
    )

    fig_sim.update_yaxes(
        autorange="reversed",
        title="Depth (m)"
    )

    fig_sim.update_xaxes(
        title="Temperature (°C)"
    )

    fig_sim.update_layout(
        height=550
    )

    st.plotly_chart(
        fig_sim,
        use_container_width=True
    )

    st.info(
        f"Changing SSH from the control panel changes the "
        f"estimated thermocline depth from the physics-inspired "
        f"model. Current simulated thermocline: "
        f"**{sim_thermocline:.1f} m**."
    )


# ============================================================
# RISK TAB
# ============================================================

with tab_risk:

    st.header(
        "🌪️ Ocean Heat & Cyclone Risk Simulator"
    )

    st.markdown("""
    Tropical cyclones can intensify when large amounts of
    warm water extend below the surface. This simulator
    demonstrates that concept using a simplified Ocean Heat
    Content index.
    """)

    r1, r2 = st.columns(2)

    with r1:

        sim_surface = st.slider(
            "Surface Temperature (°C)",
            25.0,
            32.0,
            29.0,
            0.5
        )

        sim_ssh = st.slider(
            "Sea Surface Height Anomaly (m)",
            -0.2,
            0.5,
            0.05,
            0.05
        )

        sim_wind = st.slider(
            "Background Wind (m/s)",
            0.0,
            15.0,
            5.0,
            0.5
        )

    risk_thermocline = calculate_thermocline_depth(
        sim_surface,
        sim_ssh,
        35.0,
        sim_wind
    )

    risk_profile, _ = reconstruct_vertical_profile(
        sim_surface,
        sim_ssh,
        35.0,
        sim_wind,
        STANDARD_DEPTHS
    )

    risk_ohc = calculate_ohc(
        risk_profile,
        STANDARD_DEPTHS
    )

    # Normalized demonstration score
    risk_score = np.clip(
        (
            (sim_surface - 26)
            * 12
            +
            max(sim_ssh, 0)
            * 100
            +
            risk_ohc * 0.25
        ),
        0,
        100
    )

    if risk_score < 30:

        risk_label = "LOW"
        risk_message = "Limited subsurface heat available."

    elif risk_score < 60:

        risk_label = "MODERATE"
        risk_message = "Elevated ocean heat available."

    elif risk_score < 80:

        risk_label = "HIGH"
        risk_message = "Large warm-water reservoir detected."

    else:

        risk_label = "EXTREME"
        risk_message = (
            "Very large subsurface heat reservoir."
        )


    with r2:

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={
                    "text":
                    "Subsurface Heat Risk Index"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "steps": [
                        {
                            "range": [0, 30],
                            "color": "#123d45"
                        },
                        {
                            "range": [30, 60],
                            "color": "#594c18"
                        },
                        {
                            "range": [60, 80],
                            "color": "#613b18"
                        },
                        {
                            "range": [80, 100],
                            "color": "#601b24"
                        }
                    ],
                    "threshold": {
                        "line": {
                            "color": "white",
                            "width": 4
                        },
                        "value": risk_score
                    }
                }
            )
        )

        fig_gauge.update_layout(
            height=350
        )

        st.plotly_chart(
            fig_gauge,
            use_container_width=True
        )

    st.markdown(
        f"""
        ### Current Risk: **{risk_label}**

        {risk_message}

        **Estimated thermocline:** {risk_thermocline:.1f} m

        **Relative ocean heat index:** {risk_ohc:.1f}
        """
    )

    st.warning(
        "This is an educational/PoC heat-risk indicator, "
        "not a cyclone intensity forecast. Actual cyclone "
        "forecasting requires atmospheric, oceanic and "
        "dynamical forecasting systems."
    )


# ============================================================
# ARCHITECTURE TAB
# ============================================================

with tab_architecture:

    st.header(
        "🛠️ End-to-End OceanEmbed Architecture"
    )

    st.markdown("""
    ### 1️⃣ Satellite Data Layer

    Multiple surface observations are harmonized onto a
    common spatial and temporal grid.

    **Inputs**

    - Sea Surface Temperature
    - Sea Surface Height / SLA
    - Sea Surface Salinity
    - Surface Winds
    - Ocean currents
    """)

    st.markdown("""
    ### 2️⃣ Data Harmonization

    Raw NetCDF / satellite products are:

    - quality controlled
    - interpolated
    - regridded
    - normalized
    - synchronized by time
    """)

    st.markdown("""
    ### 3️⃣ AI Embedding Engine

    Multi-channel surface fields can be passed into a
    CNN / Vision Transformer encoder.

    Conceptually:

    `Satellite Grid → Encoder → Latent Ocean State`
    """)

    st.markdown("""
    ### 4️⃣ Physics-Informed Decoder

    The latent representation is decoded into
    standardized depth levels:

    `Latent State → Decoder → T(0m ... 1000m)`
    """)

    st.markdown("""
    ### 5️⃣ ARGO Validation

    Independent ARGO observations are used to evaluate:

    - RMSE
    - Bias
    - Correlation
    - Vertical structure
    - Thermocline depth error
    """)

    st.markdown("""
    ### 6️⃣ API Layer

    A FastAPI service can expose:

    `/predict`

    `/profile`

    `/heat-risk`

    `/health`

    `/metadata`
    """)

    st.markdown("""
    ### 7️⃣ Interactive Digital Twin

    Streamlit visualizes:

    - surface maps
    - vertical profiles
    - thermocline
    - 3D temperature volumes
    - heat-risk regions
    """)

    st.divider()

    st.subheader(
        "👥 Proposed Team Responsibilities"
    )

    architecture_df = pd.DataFrame({
        "Team Member": [
            "Nithya",
            "Bhoomika",
            "Deepak",
            "Gowtham",
            "Nani",
            "Snehith"
        ],

        "Responsibility": [
            "Satellite data ingestion & preprocessing",
            "CNN / Transformer embedding engine",
            "Decoder + ARGO validation",
            "FastAPI backend",
            "Streamlit visualization",
            "Docker + system integration"
        ]
    })

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DIGITAL TWIN MAP
# ============================================================

st.divider()

st.subheader(
    "🌐 Regional Subsurface Thermal Intelligence"
)

st.caption(
    "Synthetic demonstration of predicted subsurface "
    "thermal anomalies across the Arabian Sea."
)

# Stable synthetic heat field
rng = np.random.default_rng(RNG_SEED)

radar_lats = rng.uniform(
    8.0,
    25.0,
    2500
)

radar_lons = rng.uniform(
    55.0,
    77.0,
    2500
)

intensity = (
    np.exp(
        -(
            (radar_lats - 16.5) ** 2
            +
            (radar_lons - 67.5) ** 2
        )
        / 15
    )
    * 100
)

fig_radar = px.density_map(
    lat=radar_lats,
    lon=radar_lons,
    z=intensity,
    radius=25,
    center={
        "lat": 16.0,
        "lon": 68.0
    },
    zoom=4.2,
    map_style="carto-darkmatter",
    color_continuous_scale="Inferno",
    title="Predicted Subsurface Thermal Anomaly"
)

fig_radar.update_layout(
    height=650,
    margin=dict(
        l=0,
        r=0,
        t=45,
        b=0
    )
)

st.plotly_chart(
    fig_radar,
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "OceanEmbed • Physics-Informed Subsurface Ocean "
    "Temperature Reconstruction • Prototype / PoC"
)

st.caption(
    "⚠️ Current dashboard uses synthetic data for "
    "demonstration. Replace the data-generation layer "
    "with real satellite and ARGO datasets before "
    "scientific deployment."
)