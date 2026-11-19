
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

    except (ValueError, TypeError):
        return None


# =========================================================
# DAMPING CALCULATIONS
# =========================================================

def calculate_damping(storage, loss):

    if storage <= 0 or loss < 0:
        return None, None

    tan_delta = loss / storage
    zeta = tan_delta / 2

    return tan_delta, zeta


def calculate_damping_coefficient(mass, stiffness, zeta):

    return 2 * zeta * math.sqrt(mass * stiffness)


# =========================================================
# FATIGUE CALCULATIONS
# =========================================================

def fatigue_life(calculated_tensile_strain_microstrain):

    # Nfi = Nf1 * (et1 / eti)^3.89
    # Nf1 = 98.58
    # et1 = 112.2

    nf1 = 98.58
    et1 = 112.2
    eti = calculated_tensile_strain_microstrain

    if eti <= 0 or not math.isfinite(eti):
        return None

    return nf1 * (et1 / eti) ** 3.89


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
# INTEGRAL-BASED DAMPED DISPLACEMENT
# =========================================================

def calculate_damped_displacement(
    mass,
    load,
    omega_n,
    omega_d,
    zeta,
    omega,
    simulation_time,
    steps
):

    """
    x_d(t) = (P0/M)/omega_d *
    integral from 0 to t of:

    exp[-zeta*omega_n*(t-tau)]
    * sin[omega_d*(t-tau)]
    * sin(omega*tau) d(tau)

    Evaluated at t = simulation_time.
    """

    if (
        mass <= 0
        or omega_d <= 0
        or simulation_time <= 0
        or steps < 2
    ):
        return None

    tau = np.linspace(
        0.0,
        simulation_time,
        steps
    )

    integrand = (
        np.exp(
            -zeta * omega_n * (simulation_time - tau)
        )
        * np.sin(
            omega_d * (simulation_time - tau)
        )
        * np.sin(omega * tau)
    )

    # Numerical integration

    if hasattr(np, "trapezoid"):
        integral_value = np.trapezoid(integrand, tau)
    else:
        integral_value = np.trapz(integrand, tau)

    x_damped = (
        (load / mass) / omega_d
    ) * integral_value

    return float(x_damped)


# =========================================================
# NEWMARK-BETA TRANSIENT SOLVER
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

    time = np.linspace(
        0,
        simulation_time,
        steps
    )

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

    return (
        time,
        force,
        displacement,
        velocity,
        acceleration
    )


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

    repetitions = numeric_input(
        "Applied Load Repetitions",
        "load_repetitions",
        "repetitions",
        minimum=0.000001
    )

    # Shared project parameters

    mass = float(st.session_state["mass"])
    stiffness = float(st.session_state["stiffness"])
    load = float(st.session_state["vehicle_load"])
    distance = float(st.session_state["loading_distance"])
    simulation_time = float(st.session_state["simulation_duration"])
    reference_strain = float(st.session_state["reference_strain"])
    steps = int(float(st.session_state["time_steps"]))

    st.divider()

    st.subheader("Calculated Results")

    if (
        storage is not None
        and loss is not None
        and vehicle_speed is not None
        and repetitions is not None
    ):

        tan_delta, zeta = calculate_damping(
            storage,
            loss
        )

        if zeta >= 1:

            st.error(
                "Damping ratio must be below 1 for this underdamped model."
            )

        elif (
            mass <= 0
            or stiffness <= 0
            or load <= 0
            or distance <= 0
        ):

            st.error("Project parameters must be positive.")

        elif steps < 100 or steps > 100000:

            st.error(
                "Time steps must be between 100 and 100000."
            )

        else:

            # -------------------------------------
            # DAMPING COEFFICIENT
            # -------------------------------------

            damping_c = calculate_damping_coefficient(
                mass,
                stiffness,
                zeta
            )

            # -------------------------------------
            # NATURAL FREQUENCIES
            # -------------------------------------

            omega_n = math.sqrt(
                stiffness / mass
            )

            natural_frequency = omega_n / (2 * math.pi)

            omega_d = omega_n * math.sqrt(
                1 - zeta**2
            )

            damped_frequency = omega_d / (2 * math.pi)

            # -------------------------------------
            # VEHICLE LOADING
            # -------------------------------------

            speed_ms, td, fc, omega = moving_load_parameters(
                vehicle_speed,
                distance,
                load
            )

            # -------------------------------------
            # NEWMARK-BETA RESPONSE
            # -------------------------------------

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

                (
                    time,
                    force,
                    displacement,
                    velocity,
                    acceleration
                ) = response

                dynamic_peak = np.max(
                    np.abs(displacement)
                )

                # -------------------------------------
                # STATIC DISPLACEMENT
                # -------------------------------------

                static_displacement = load / stiffness

                # -------------------------------------
                # DYNAMIC AMPLIFICATION FACTOR
                # -------------------------------------

                # DAF = 1 / sqrt[(1-r²)² + (2*zeta*r)²]
                # r = f / f_n

                frequency_ratio = (
                    fc / natural_frequency
                )

                daf_denominator = math.sqrt(
                    (1 - frequency_ratio**2)**2
                    + (2 * zeta * frequency_ratio)**2
                )

                daf = (
                    1 / daf_denominator
                    if daf_denominator > 0
                    else None
                )

                # -------------------------------------
                # DAMPED DISPLACEMENT
                # INTEGRAL-BASED FORMULA
                # -------------------------------------

                damped_displacement = calculate_damped_displacement(
                    mass=mass,
                    load=load,
                    omega_n=omega_n,
                    omega_d=omega_d,
                    zeta=zeta,
                    omega=omega,
                    simulation_time=simulation_time,
                    steps=steps
                )

                # -------------------------------------
                # UNDAMPED DISPLACEMENT
                # AT SIMULATION DURATION
                # -------------------------------------

                t_eval = simulation_time

                denominator = (
                    omega_n**2 - omega**2
                )

                resonance_tolerance = (
                    1e-10
                    * max(omega_n**2, omega**2, 1.0)
                )

                if abs(denominator) <= resonance_tolerance:

                    # Limit as omega approaches omega_n

                    undamped_displacement = (
                        (load / mass)
                        * (
                            math.sin(omega_n * t_eval)
                            - omega_n * t_eval
                            * math.cos(omega_n * t_eval)
                        )
                        / (2 * omega_n**2)
                    )

                else:

                    undamped_displacement = (
                        (load / mass) / denominator
                        * (
                            math.sin(omega * t_eval)
                            - (omega / omega_n)
                            * math.sin(omega_n * t_eval)
                        )
                    )

                # -------------------------------------
                # RESPONSE FACTOR
                # -------------------------------------

                if damped_displacement is None:

                    st.error(
                        "Damped displacement could not be calculated."
                    )

                elif abs(undamped_displacement) <= 1e-15:

                    st.error(
                        "Undamped displacement is approximately zero "
                        "at the simulation duration. Response Factor "
                        "cannot be calculated."
                    )

                else:

                    # RF = X_damped / X_undamped

                    response_factor = (
                        abs(damped_displacement)
                        / abs(undamped_displacement)
                    )

                    reduction = (
                        1 - response_factor
                    ) * 100

                    # -------------------------------------
                    # TENSILE STRAIN
                    # -------------------------------------

                    # epsilon_damped = RF * epsilon_static

                    tensile_strain = (
                        reference_strain
                        * response_factor
                        * 1e-6
                    )

                    tensile_strain_micro = (
                        tensile_strain * 1e6
                    )

                    # -------------------------------------
                    # FATIGUE LIFE
                    # -------------------------------------

                    nf = fatigue_life(
                        tensile_strain_micro
                    )

                    # -------------------------------------
                    # FATIGUE DAMAGE
                    # -------------------------------------

                    damage = fatigue_damage(
                        repetitions,
                        nf
                    )

                    # =====================================
                    # DISPLAY RESULTS
                    # =====================================

                    st.markdown(
                        "### 1. Damping Properties"
                    )

                    a, b, c = st.columns(3)

                    a.metric(
                        "tan(δ)",
                        f"{tan_delta:.6f}"
                    )

                    b.metric(
                        "Damping Ratio (ζ)",
                        f"{zeta:.6f}"
                    )

                    c.metric(
                        "Damping Coefficient (C)",
                        f"{damping_c:,.4f} Ns/m"
                    )

                    # -------------------------------------

                    st.divider()

                    st.markdown(
                        "### 2. Dynamic Amplification Factor"
                    )

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
                        f"{daf:.6f}" if daf is not None else "—"
                    )

                    # -------------------------------------

                    st.divider()

                    st.markdown(
                        "### 3. Response Factor"
                    )

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

                    st.divider()

                    st.markdown(
                        "### 4. Tensile Strain Calculation"
                    )

                    a, b = st.columns(2)

                    a.metric(
                        "Reference Strain",
                        f"{reference_strain:.4f} με"
                    )

                    b.metric(
                        "Calculated Tensile Strain",
                        f"{tensile_strain_micro:.4f} με"
                    )

                    # -------------------------------------

                    st.divider()

                    st.markdown(
                        "### 5. Fatigue Damage"
                    )

                    st.metric(
                        "Fatigue Damage",
                        f"{damage:.8f}"
                        if damage is not None
                        else "—"
                    )

                    # =====================================
                    # COMPLETE RESULTS TABLE
                    # =====================================

                    st.divider()

                    st.subheader(
                        "Complete Calculation Results"
                    )

                    results = pd.DataFrame([
                        ["Storage Modulus", storage, "MPa"],
                        ["Loss Modulus", loss, "MPa"],
                        ["tan(δ)", tan_delta, "-"],
                        ["Damping Ratio", zeta, "-"],
                        ["Damping Coefficient", damping_c, "Ns/m"],
                        [
                            "Damped Displacement at Simulation Duration",
                            damped_displacement,
                            "m"
                        ],
                        [
                            "Undamped Displacement at Simulation Duration",
                            undamped_displacement,
                            "m"
                        ],
                        [
                            "Static Displacement",
                            static_displacement,
                            "m"
                        ],
                        [
                            "Frequency Ratio (r)",
                            frequency_ratio,
                            "-"
                        ],
                        [
                            "Dynamic Amplification Factor",
                            daf,
                            "-"
                        ],
                        [
                            "Response Factor",
                            response_factor,
                            "-"
                        ],
                        [
                            "Response Reduction",
                            reduction,
                            "%"
                        ],
                        [
                            "Reference Tensile Strain",
                            reference_strain,
                            "με"
                        ],
                        [
                            "Calculated Tensile Strain",
                            tensile_strain_micro,
                            "με"
                        ],
                        [
                            "Applied Load Repetitions",
                            repetitions,
                            "repetitions"
                        ],
                        [
                            "Fatigue Damage",
                            damage,
                            "-"
                        ]
                    ], columns=[
                        "Parameter",
                        "Value",
                        "Unit"
                    ])

                    st.dataframe(
                        results,
                        hide_index=True,
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Calculation Results",
                        results.to_csv(
                            index=False
                        ).encode("utf-8"),
                        "pavement_results.csv",
                        "text/csv",
                        key="download_results"
                    )

            else:

                st.error(
                    "Transient response calculation failed. "
                    "Check the project parameters."
                )

    else:

        st.write(
            "Enter valid material properties, vehicle speed, "
            "and repetitions."
        )


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

    # Numerical time steps = 4000 by default.
    # Hidden from the interface but used by the calculations.

    st.divider()

    st.subheader("Current Project Values")

    parameter_data = [
        ["Mass", st.session_state["mass"], "kg"],
        ["Stiffness", st.session_state["stiffness"], "N/m"],
        ["Vehicle Load", st.session_state["vehicle_load"], "N"],
        ["Loading Distance", st.session_state["loading_distance"], "m"],
        [
            "Simulation Duration",
            st.session_state["simulation_duration"],
            "s"
        ],
        [
            "Temperature",
            st.session_state["temperature"],
            "°C"
        ],
        [
            "Reference Tensile Strain",
            st.session_state["reference_strain"],
            "με"
        ],
        [
            "Asphalt Modulus",
            st.session_state["asphalt_modulus"],
            "psi"
        ],
        [
            "Vehicle Speed",
            st.session_state["vehicle_speed"],
            "km/h"
        ]
    ]

    st.dataframe(
        pd.DataFrame(
            parameter_data,
            columns=[
                "Parameter",
                "Value",
                "Unit"
            ]
        ),
        hide_index=True,
        use_container_width=True
    )
