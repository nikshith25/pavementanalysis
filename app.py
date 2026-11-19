import math
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Flexible Pavement Analysis",
    page_icon="🏗️",
    layout="wide"
)

st.title("Flexible Pavement Response & Fatigue Analysis")


# =====================================================
# DEFAULT VALUES — INITIALISE ONLY ONCE
# =====================================================

DEFAULTS = {
    "storage_modulus": "8043",
    "loss_modulus": "3184",
    "mass": "264",
    "stiffness": "1.479915e9",
    "vehicle_load": "560000",
    "loading_distance": "0.2132",
    "simulation_duration": "0.15",
    "temperature": "21.1",
    "vehicle_speeds": [20, 40, 60, 80]
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# NUMERIC INPUT FUNCTION
# =====================================================

def numeric_input(label, key, unit="", minimum=None):

    label_display = f"{label} ({unit})" if unit else label

    raw = st.text_input(
        label_display,
        key=key
    ).strip().replace(",", "")

    if raw == "":
        return None

    try:
        value = float(raw)

        if not math.isfinite(value):
            return None

        if minimum is not None and value < minimum:
            st.caption(f"{label} must be at least {minimum}.")
            return None

        return value

    except ValueError:
        st.caption(f"Enter a valid number for {label}.")
        return None


# =====================================================
# CALCULATION FUNCTIONS
# =====================================================

def calculate_damping(storage, loss):

    if storage is None or loss is None:
        return None, None

    if storage <= 0 or loss < 0:
        return None, None

    tan_delta = loss / storage
    zeta = tan_delta / 2

    return tan_delta, zeta


def calculate_damping_coefficient(mass, stiffness, zeta):

    if mass is None or stiffness is None or zeta is None:
        return None

    if mass <= 0 or stiffness <= 0:
        return None

    return 2 * zeta * math.sqrt(mass * stiffness)


def calculate_daf(r, zeta):

    denominator = math.sqrt(
        (1 - r**2)**2 + (2 * zeta * r)**2
    )

    if denominator == 0:
        return None

    return 1 / denominator


# =====================================================
# READ SHARED PROJECT VALUES
# =====================================================

def get_project_values():

    mass = numeric_input(
        "Mass (M)",
        "mass",
        "kg",
        minimum=0.000001
    )

    stiffness = numeric_input(
        "Stiffness (K)",
        "stiffness",
        "N/m",
        minimum=0.000001
    )

    return mass, stiffness


# =====================================================
# MAIN TABS
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "Damping Calculator",
    "Project Parameters",
    "Visualisation"
])


# =====================================================
# TAB 1 — DAMPING CALCULATOR
# =====================================================

with tab1:

    st.subheader("Material Properties")

    col1, col2 = st.columns(2)

    with col1:

        storage = numeric_input(
            "Storage Modulus (E′)",
            "storage_modulus",
            "MPa",
            minimum=0.000001
        )

    with col2:

        loss = numeric_input(
            "Loss Modulus (E″)",
            "loss_modulus",
            "MPa",
            minimum=0
        )

    # Read shared project parameters.
    # These are edited only in Tab 2.

    mass = float(st.session_state["mass"])
    stiffness = float(st.session_state["stiffness"])

    tan_delta, zeta = calculate_damping(
        storage,
        loss
    )

    st.divider()

    st.subheader("Calculated Results")

    if tan_delta is not None:

        damping_c = calculate_damping_coefficient(
            mass,
            stiffness,
            zeta
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "tan(δ)",
            f"{tan_delta:.6f}"
        )

        r2.metric(
            "Damping Ratio (ζ)",
            f"{zeta:.6f}"
        )

        r3.metric(
            "Damping Coefficient (C)",
            f"{damping_c:,.4f} Ns/m"
        )

        st.divider()

        results = pd.DataFrame([
            ["Storage Modulus", storage, "MPa"],
            ["Loss Modulus", loss, "MPa"],
            ["tan(δ)", tan_delta, "-"],
            ["Damping Ratio", zeta, "-"],
            ["Mass", mass, "kg"],
            ["Stiffness", stiffness, "N/m"],
            ["Damping Coefficient", damping_c, "Ns/m"]
        ], columns=["Parameter", "Value", "Unit"])

        st.dataframe(
            results,
            hide_index=True,
            use_container_width=True
        )

    else:

        st.write("Enter valid storage and loss modulus values.")


# =====================================================
# TAB 2 — PROJECT PARAMETERS
# =====================================================

with tab2:

    st.subheader("Project Parameters")

    st.write(
        "All reference values are prefilled. "
        "You can edit any parameter here."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        mass = numeric_input(
            "Mass (M)",
            "mass",
            "kg",
            minimum=0.000001
        )

        stiffness = numeric_input(
            "Stiffness (K)",
            "stiffness",
            "N/m",
            minimum=0.000001
        )

        vehicle_load = numeric_input(
            "Vehicle Load (P₀)",
            "vehicle_load",
            "N",
            minimum=0
        )

    with col2:

        loading_distance = numeric_input(
            "Loading Distance (D)",
            "loading_distance",
            "m",
            minimum=0.000001
        )

        simulation_duration = numeric_input(
            "Simulation Duration",
            "simulation_duration",
            "s",
            minimum=0.000001
        )

        temperature = numeric_input(
            "Temperature",
            "temperature",
            "°C"
        )

    with col3:

        speeds = st.multiselect(
            "Vehicle Speeds",
            options=[20, 40, 60, 80],
            key="vehicle_speeds"
        )

    st.divider()

    st.subheader("Current Material Inputs")

    storage = float(st.session_state["storage_modulus"])
    loss = float(st.session_state["loss_modulus"])

    st.dataframe(
        pd.DataFrame([
            ["Storage Modulus", storage, "MPa"],
            ["Loss Modulus", loss, "MPa"]
        ], columns=["Parameter", "Value", "Unit"]),
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.subheader("Calculated Reference Results")

    tan_delta, zeta = calculate_damping(
        storage,
        loss
    )

    if zeta is not None and mass is not None and stiffness is not None:

        damping_c = calculate_damping_coefficient(
            mass,
            stiffness,
            zeta
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Calculated Damping Ratio",
            f"{zeta:.6f}"
        )

        c2.metric(
            "Calculated Damping Coefficient",
            f"{damping_c:,.4f} Ns/m"
        )


# =====================================================
# TAB 3 — VISUALISATION
# =====================================================

with tab3:

    st.subheader("Result Visualisation")

    storage = float(st.session_state["storage_modulus"])
    loss = float(st.session_state["loss_modulus"])

    mass = float(st.session_state["mass"])
    stiffness = float(st.session_state["stiffness"])

    tan_delta, zeta = calculate_damping(
        storage,
        loss
    )

    if zeta is not None:

        damping_c = calculate_damping_coefficient(
            mass,
            stiffness,
            zeta
        )

        # ---------------------------------------------
        # MODULUS GRAPH
        # ---------------------------------------------

        if st.checkbox(
            "Visualise Storage and Loss Modulus",
            key="visual_modulus"
        ):

            fig1 = go.Figure(
                go.Bar(
                    x=["Storage Modulus", "Loss Modulus"],
                    y=[storage, loss],
                    text=[
                        f"{storage:.2f}",
                        f"{loss:.2f}"
                    ],
                    textposition="outside"
                )
            )

            fig1.update_layout(
                template="plotly_white",
                title="Storage and Loss Modulus",
                xaxis_title="Material Property",
                yaxis_title="Modulus (MPa)"
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

        # ---------------------------------------------
        # DAMPING RATIO GRAPH
        # ---------------------------------------------

        if st.checkbox(
            "Visualise Damping Ratio",
            key="visual_damping"
        ):

            fig2 = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=zeta,
                    title={"text": "Damping Ratio (ζ)"},
                    gauge={
                        "axis": {
                            "range": [0, max(0.5, zeta * 1.25)]
                        }
                    }
                )
            )

            fig2.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        # ---------------------------------------------
        # DAMPING COEFFICIENT
        # ---------------------------------------------

        if st.checkbox(
            "Visualise Damping Coefficient",
            key="visual_coefficient"
        ):

            fig3 = go.Figure(
                go.Bar(
                    x=["Damping Coefficient"],
                    y=[damping_c],
                    text=[f"{damping_c:.4f}"],
                    textposition="outside"
                )
            )

            fig3.update_layout(
                template="plotly_white",
                title="Calculated Damping Coefficient",
                yaxis_title="C (Ns/m)"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )


# =====================================================
# EXPORT RESULTS
# =====================================================

st.divider()

st.subheader("Export Results")

try:

    storage = float(st.session_state["storage_modulus"])
    loss = float(st.session_state["loss_modulus"])
    mass = float(st.session_state["mass"])
    stiffness = float(st.session_state["stiffness"])

    tan_delta, zeta = calculate_damping(
        storage,
        loss
    )

    if zeta is not None:

        damping_c = calculate_damping_coefficient(
            mass,
            stiffness,
            zeta
        )

        export_df = pd.DataFrame([
            ["Storage Modulus", storage, "MPa"],
            ["Loss Modulus", loss, "MPa"],
            ["tan(δ)", tan_delta, "-"],
            ["Damping Ratio", zeta, "-"],
            ["Mass", mass, "kg"],
            ["Stiffness", stiffness, "N/m"],
            ["Damping Coefficient", damping_c, "Ns/m"]
        ], columns=["Parameter", "Value", "Unit"])

        st.download_button(
            "Download Results CSV",
            export_df.to_csv(index=False).encode("utf-8"),
            file_name="pavement_damping_results.csv",
            mime="text/csv",
            key="download_results"
        )

except (ValueError, TypeError):
    pass

