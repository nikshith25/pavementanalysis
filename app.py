
import math
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Flexible Pavement Analysis",
    page_icon="🏗️",
    layout="wide"
)

st.title("Flexible Pavement Response & Fatigue Analysis")


# =========================================================
# DEFAULT PARAMETERS
# =========================================================

DEFAULTS = {
    "storage_modulus": "8043",
    "loss_modulus": "3184",
    "vehicle_speed": "40",
    "mass": "264",
    "stiffness": "1.479915e9",
    "vehicle_load": "560000",
    "loading_distance": "0.2132",
    "simulation_duration": "0.15",
    "temperature": "21.1",
    "reference_strain": "112.2",
    "asphalt_modulus": "300000",
    "load_repetitions": "1000000",
    "time_steps": "4000"
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# NUMERIC INPUT
# =========================================================

def numeric_input(label, key, unit="", minimum=None):

    label_text = f"{label} ({unit})" if unit else label

    raw = st.text_input(
        label_text,
        key=key
    ).strip().replace(",", "")

    if not raw:
        return None

    try:
        value = float(raw)

        if not math.isfinite(value):
            return None

        if minimum is not None and value < minimum:
            return None

        return value

    except ValueError:
        return None


# =========================================================
# CALCULATION FUNCTIONS
# =========================================================

def calculate_damping(storage, loss):

    if storage <= 0 or loss < 0:
        return None, None

    tan_delta = loss / storage
    zeta = tan_delta / 2

    return tan_delta, zeta


def calculate_damping_coefficient(mass, stiffness, zeta):

    return 2 * zeta * math.sqrt(mass * stiffness)


def fatigue_life(strain, modulus):

    if strain <= 0 or modulus <= 0:
        return None

    return (
        0.0796
        * (1 / strain) ** 3.291
        * (1 / modulus) ** 0.854
    )


def fatigue_damage(repetitions, allowable_life):

    if allowable_life is None or allowable_life <= 0:
        return None

    return repetitions / allowable_life


# =========================================================
# MOVING VEHICLE LOAD
# =========================================================

def moving_load_parameters(speed_kmh, distance, load):

    speed_ms = speed_kmh / 3.6

    duration = distance / speed_ms

    characteristic_frequency = 1 / (2 * duration)

    angular_frequency = 2 * math.pi * characteristic_frequency

    return (
        speed_ms,
        duration,
        characteristic_frequency,
        angular_frequency
    )


# =========================================================
# NEWMARK-BETA SDOF TRANSIENT SOLVER
# =========================================================

def solve_moving_load(
    mass,
    stiffness,
    damping,
    load,
    duration,
    simulation_time,
    steps
):

    if min(mass, stiffness, duration, simulation_time) <= 0:
        return None

    dt = simulation_time / (steps - 1)

    time = np.linspace(0, simulation_time, steps)

    # Half-sine moving-load pulse
    force = np.zeros(steps)

    active = time <= duration

    force[active] = load * np.sin(
        math.pi * time[active] / duration
    )

    # Newmark average acceleration parameters
    beta = 0.25
    gamma = 0.5

    a0 = 1 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = dt * (gamma / (2 * beta) - 1)

    effective_stiffness = (
        stiffness + a0 * mass + a1 * damping
    )

    displacement = np.zeros(steps)
    velocity = np.zeros(steps)
    acceleration = np.zeros(steps)

    acceleration[0] = (
        force[0]
        - damping * velocity[0]
        - stiffness * displacement[0]
    ) / mass

    for i in range(1, steps):

        effective_force = (
            force[i]
            + mass * (
                a0 * displacement[i-1]
                + a2 * velocity[i-1]
                + a3 * acceleration[i-1]
            )
            + damping * (
                a1 * displacement[i-1]
                + a4 * velocity[i-1]
                + a5 * acceleration[i-1]
            )
        )

        displacement[i] = (
            effective_force / effective_stiffness
        )

        acceleration[i] = (
            a0 * (displacement[i] - displacement[i-1])
            - a2 * velocity[i-1]
            - a3 * acceleration[i-1]
        )

        velocity[i] = (
            velocity[i-1]
            + dt * (
                (1-gamma) * acceleration[i-1]
                + gamma * acceleration[i]
            )
        )

    return time, force, displacement, velocity, acceleration


# =========================================================
# MAIN TABS
# =========================================================

tab1, tab2 = st.tabs([
    "Damping & Response Calculator",
    "Project Parameters"
])


# =========================================================
# TAB 1 — MAIN CALCULATOR
# =========================================================

with tab1:

    st.subheader("Material Properties")

    c1, c2 = st.columns(2)

    with c1:
        storage = numeric_input(
            "Storage Modulus (E′)",
            "storage_modulus",
            "MPa",
            minimum=0.000001
        )

    with c2:
        loss = numeric_input(
            "Loss Modulus (E″)",
            "loss_modulus",
            "MPa",
            minimum=0
        )

    st.divider()

    st.subheader("Vehicle Loading")

    vehicle_speed = numeric_input(
        "Vehicle Speed",
        "vehicle_speed",
        "km/h",
        minimum=0.000001
    )

    # Shared project parameters

    mass = float(st.session_state["mass"])
    stiffness = float(st.session_state["stiffness"])
    load = float(st.session_state["vehicle_load"])
    distance = float(st.session_state["loading_distance"])
    simulation_time = float(st.session_state["simulation_duration"])
    reference_strain = float(st.session_state["reference_strain"])
    asphalt_modulus = float(st.session_state["asphalt_modulus"])
    repetitions = float(st.session_state["load_repetitions"])
    steps = int(float(st.session_state["time_steps"]))

    st.divider()

    st.subheader("Calculated Results")

    if (
        storage is not None
        and loss is not None
        and vehicle_speed is not None
    ):

        tan_delta, zeta = calculate_damping(
            storage,
            loss
        )

        if zeta >= 1:

            st.error(
                "Damping ratio must be below 1 for this underdamped model."
            )

        elif mass <= 0 or stiffness <= 0 or load <= 0 or distance <= 0:

            st.error("Project parameters must be positive.")

        elif steps < 100 or steps > 100000:

            st.error("Time steps must be between 100 and 100000.")

        else:

            damping_c = calculate_damping_coefficient(
                mass,
                stiffness,
                zeta
            )

            # Natural frequency

            omega_n = math.sqrt(stiffness / mass)

            natural_frequency = omega_n / (2 * math.pi)

            omega_d = omega_n * math.sqrt(1 - zeta**2)

            damped_frequency = omega_d / (2 * math.pi)

            # Vehicle loading parameters

            speed_ms, td, fc, omega = moving_load_parameters(
                vehicle_speed,
                distance,
                load
            )

            # Transient response

            response = solve_moving_load(
                mass,
                stiffness,
                damping_c,
                load,
                td,
                simulation_time,
                steps
            )

            if response is not None:

                time, force, displacement, velocity, acceleration = response

                dynamic_peak = np.max(np.abs(displacement))

                static_displacement = load / stiffness

                if static_displacement > 0:

                    daf = dynamic_peak / static_displacement

                    # Response factor uses dynamic displacement
                    # divided by static reference displacement.

                    response_factor = daf

                    reduction = (1 - response_factor) * 100

                    # Preliminary strain sensitivity:
                    # strain is scaled by displacement DAF.

                    tensile_strain = (
                        reference_strain
                        * response_factor
                        * 1e-6
                    )

                    nf = fatigue_life(
                        tensile_strain,
                        asphalt_modulus
                    )

                    damage = fatigue_damage(
                        repetitions,
                        nf
                    )

                    # -------------------------------------
                    # 1. DAMPING RESULTS
                    # -------------------------------------

                    st.markdown("### 1. Damping Properties")

                    a, b, c = st.columns(3)

                    a.metric("tan(δ)", f"{tan_delta:.6f}")

                    b.metric("Damping Ratio (ζ)", f"{zeta:.6f}")

                    c.metric(
                        "Damping Coefficient (C)",
                        f"{damping_c:,.4f}"
                    )

                    # -------------------------------------
                    # 2. VEHICLE LOADING
                    # -------------------------------------

                    st.divider()

                    st.markdown("### 2. Moving Vehicle Load")

                    a, b, c, d = st.columns(4)

                    a.metric("Speed", f"{vehicle_speed:.3f} km/h")

                    b.metric("Speed", f"{speed_ms:.4f} m/s")

                    c.metric("Loading Duration", f"{td:.6f} s")

                    d.metric("Characteristic Frequency", f"{fc:.4f} Hz")

                    a, b = st.columns(2)

                    a.metric(
                        "Angular Frequency (ω)",
                        f"{omega:.4f} rad/s"
                    )

                    b.metric(
                        "Loading Distance",
                        f"{distance:.4f} m"
                    )

                    # -------------------------------------
                    # 3. NATURAL FREQUENCY
                    # -------------------------------------

                    st.divider()

                    st.markdown("### 3. Natural Frequency")

                    a, b, c = st.columns(3)

                    a.metric(
                        "Undamped Natural Frequency",
                        f"{natural_frequency:.4f} Hz"
                    )

                    b.metric(
                        "Damped Natural Frequency",
                        f"{damped_frequency:.4f} Hz"
                    )

                    c.metric(
                        "Natural Angular Frequency",
                        f"{omega_n:.4f} rad/s"
                    )

                    # -------------------------------------
                    # 4. DYNAMIC AMPLIFICATION FACTOR
                    # -------------------------------------

                    st.divider()

                    st.markdown("### 4. Dynamic Amplification Factor")

                    a, b, c = st.columns(3)

                    a.metric(
                        "Dynamic Peak Displacement",
                        f"{dynamic_peak * 1000:.6f} mm"
                    )

                    b.metric(
                        "Static Displacement",
                        f"{static_displacement * 1000:.6f} mm"
                    )

                    c.metric(
                        "Dynamic Amplification Factor",
                        f"{daf:.6f}"
                    )

                    # -------------------------------------
                    # 5. RESPONSE FACTOR
                    # -------------------------------------

                    st.divider()

                    st.markdown("### 5. Response Factor")

                    a, b = st.columns(2)

                    a.metric(
                        "Response Factor",
                        f"{response_factor:.6f}"
                    )

                    b.metric(
                        "Response Reduction",
                        f"{reduction:.4f}%"
                    )

                    # -------------------------------------
                    # 6. STRAIN CALCULATION
                    # -------------------------------------

                    st.divider()

                    st.markdown("### 6. Tensile Strain Calculation")

                    a, b = st.columns(2)

                    a.metric(
                        "Reference Strain",
                        f"{reference_strain:.4f} με"
                    )

                    b.metric(
                        "Calculated Tensile Strain",
                        f"{tensile_strain * 1e6:.4f} με"
                    )

                    # -------------------------------------
                    # 7. FATIGUE DAMAGE
                    # -------------------------------------

                    st.divider()

                    st.markdown("### 7. Fatigue Damage")

                    a, b, c = st.columns(3)

                    a.metric(
                        "Allowable Fatigue Life (Nf)",
                        f"{nf:,.2f}" if nf else "—"
                    )

                    b.metric(
                        "Applied Repetitions",
                        f"{repetitions:,.0f}"
                    )

                    c.metric(
                        "Fatigue Damage",
                        f"{damage:.8f}" if damage is not None else "—"
                    )

                    # -------------------------------------
                    # COMPLETE RESULTS
                    # -------------------------------------

                    st.divider()

                    st.subheader("Complete Calculation Results")

                    results = pd.DataFrame([
                        ["Storage Modulus", storage, "MPa"],
                        ["Loss Modulus", loss, "MPa"],
                        ["tan(δ)", tan_delta, "-"],
                        ["Damping Ratio", zeta, "-"],
                        ["Damping Coefficient", damping_c, "Ns/m"],
                        ["Vehicle Speed", vehicle_speed, "km/h"],
                        ["Vehicle Speed", speed_ms, "m/s"],
                        ["Loading Duration", td, "s"],
                        ["Characteristic Frequency", fc, "Hz"],
                        ["Angular Frequency", omega, "rad/s"],
                        ["Natural Frequency", natural_frequency, "Hz"],
                        ["Damped Natural Frequency", damped_frequency, "Hz"],
                        ["Dynamic Peak Displacement", dynamic_peak, "m"],
                        ["Static Displacement", static_displacement, "m"],
                        ["Dynamic Amplification Factor", daf, "-"],
                        ["Response Factor", response_factor, "-"],
                        ["Response Reduction", reduction, "%"],
                        ["Reference Tensile Strain", reference_strain, "με"],
                        ["Calculated Tensile Strain", tensile_strain * 1e6, "με"],
                        ["Fatigue Life", nf, "repetitions"],
                        ["Applied Repetitions", repetitions, "repetitions"],
                        ["Fatigue Damage", damage, "-"]
                    ], columns=["Parameter", "Value", "Unit"])

                    st.dataframe(
                        results,
                        hide_index=True,
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Calculation Results",
                        results.to_csv(index=False).encode("utf-8"),
                        "pavement_results.csv",
                        "text/csv",
                        key="download_results"
                    )

    else:
        st.write("Enter valid material properties and vehicle speed.")


# =========================================================
# TAB 2 — PROJECT PARAMETERS
# =========================================================

with tab2:

    st.subheader("Project Parameters")

    st.write(
        "All default values are prefilled and editable. "
        "Changes automatically update the main calculation."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        mass = numeric_input(
            "Equivalent Mass",
            "mass",
            "kg",
            minimum=0.000001
        )

        stiffness = numeric_input(
            "Equivalent Stiffness",
            "stiffness",
            "N/m",
            minimum=0.000001
        )

        vehicle_load = numeric_input(
            "Vehicle Load",
            "vehicle_load",
            "N",
            minimum=0.000001
        )

        loading_distance = numeric_input(
            "Characteristic Loading Length",
            "loading_distance",
            "m",
            minimum=0.000001
        )

    with c2:

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

        reference_strain = numeric_input(
            "Reference Tensile Strain",
            "reference_strain",
            "με",
            minimum=0.000001
        )

        asphalt_modulus = numeric_input(
            "Asphalt Modulus",
            "asphalt_modulus",
            "psi",
            minimum=0.000001
        )

    with c3:

        repetitions = numeric_input(
            "Applied Load Repetitions",
            "load_repetitions",
            "repetitions",
            minimum=0
        )

        steps = numeric_input(
            "Numerical Time Steps",
            "time_steps",
            minimum=100
        )

    st.divider()

    st.subheader("Current Project Values")

    parameter_data = [
        ["Mass", st.session_state["mass"], "kg"],
        ["Stiffness", st.session_state["stiffness"], "N/m"],
        ["Vehicle Load", st.session_state["vehicle_load"], "N"],
        ["Loading Distance", st.session_state["loading_distance"], "m"],
        ["Simulation Duration", st.session_state["simulation_duration"], "s"],
        ["Temperature", st.session_state["temperature"], "°C"],
        ["Reference Tensile Strain", st.session_state["reference_strain"], "με"],
        ["Asphalt Modulus", st.session_state["asphalt_modulus"], "psi"],
        ["Applied Load Repetitions", st.session_state["load_repetitions"], "repetitions"],
        ["Vehicle Speed", st.session_state["vehicle_speed"], "km/h"]
    ]

    st.dataframe(
        pd.DataFrame(
            parameter_data,
            columns=["Parameter", "Value", "Unit"]
        ),
        hide_index=True,
        use_container_width=True
    )
