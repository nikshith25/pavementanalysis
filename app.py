
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


def get_project_value(key):

    try:
        value = float(st.session_state[key])

        if not math.isfinite(value):
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

    # Damping ratio
    zeta = tan_delta / 2

    return tan_delta, zeta


def calculate_damping_coefficient(mass, stiffness, zeta):

    return 2 * zeta * math.sqrt(mass * stiffness)


# =========================================================
# NATURAL FREQUENCY
# =========================================================

def calculate_natural_frequency(mass, stiffness):

    if mass <= 0 or stiffness <= 0:
        return None, None

    # Natural angular frequency
    omega_n = math.sqrt(stiffness / mass)

    # Natural frequency
    f_n = omega_n / (2 * math.pi)

    return omega_n, f_n


# =========================================================
# MOVING VEHICLE LOAD
# =========================================================

def moving_load_parameters(speed_kmh, distance):

    if speed_kmh <= 0 or distance <= 0:
        return None, None, None, None

    # Convert vehicle speed from km/h to m/s
    speed_ms = speed_kmh / 3.6

    # Time taken by vehicle to cross loading distance
    loading_duration = distance / speed_ms

    # USER-SPECIFIED FREQUENCY FORMULA
    # f = (v / 3.6) / D

    frequency = speed_ms / distance

    # Angular frequency
    omega = 2 * math.pi * frequency

    return (
        speed_ms,
        loading_duration,
        frequency,
        omega
    )


# =========================================================
# DYNAMIC AMPLIFICATION FACTOR
# =========================================================

def calculate_daf(frequency, natural_frequency, zeta):

    if natural_frequency <= 0:
        return None

    # Frequency ratio
    r = frequency / natural_frequency

    # DAF = 1 / sqrt[(1-r²)² + (2*zeta*r)²]

    denominator = math.sqrt(
        (1 - r**2)**2
        + (2 * zeta * r)**2
    )

    if denominator <= 0:
        return None

    daf = 1 / denominator

    return r, daf


# =========================================================
# STATIC AND DAMPED DISPLACEMENT
# =========================================================

def calculate_static_displacement(load, stiffness):

    if stiffness <= 0:
        return None

    return load / stiffness


def calculate_damped_displacement(daf, static_displacement):

    if daf is None or static_displacement is None:
        return None

    # Damped displacement amplitude
    # X_damped = DAF * X_static

    return daf * static_displacement


# =========================================================
# UNDAMPED DISPLACEMENT
# =========================================================

def calculate_undamped_displacement(
    mass,
    load,
    omega_n,
    omega,
    simulation_time
):

    if (
        mass <= 0
        or omega_n <= 0
        or simulation_time <= 0
    ):
        return None

    t = simulation_time

    denominator = omega_n**2 - omega**2

    tolerance = (
        1e-10
        * max(omega_n**2, omega**2, 1.0)
    )

    if abs(denominator) <= tolerance:

        # Resonance limit as omega approaches omega_n

        displacement = (
            (load / mass)
            * (
                math.sin(omega_n * t)
                - omega_n * t * math.cos(omega_n * t)
            )
            / (2 * omega_n**2)
        )

    else:

        # User-specified undamped displacement equation

        displacement = (
            (load / mass) / denominator
            * (
                math.sin(omega * t)
                - (omega / omega_n)
                * math.sin(omega_n * t)
            )
        )

    return displacement


# =========================================================
# RESPONSE FACTOR
# =========================================================

def calculate_response_factor(
    damped_displacement,
    undamped_displacement
):

    if damped_displacement is None:
        return None

    if undamped_displacement is None:
        return None

    if abs(undamped_displacement) <= 1e-15:
        return None

    # RF = |X_damped| / |X_undamped|

    return (
        abs(damped_displacement)
        / abs(undamped_displacement)
    )


# =========================================================
# TENSILE STRAIN
# =========================================================

def calculate_tensile_strain(
    response_factor,
    reference_strain
):

    if response_factor is None:
        return None

    # Reference strain is already in microstrain.
    # epsilon_damped = RF * epsilon_static

    tensile_strain_micro = (
        response_factor * reference_strain
    )

    return tensile_strain_micro


# =========================================================
# FATIGUE CALCULATIONS
# =========================================================

def fatigue_life(calculated_tensile_strain_microstrain):

    # Nfi = 98.58 * (112.2 / eti)^3.89

    nf1 = 98.58
    et1 = 112.2

    eti = calculated_tensile_strain_microstrain

    if eti is None or eti <= 0:
        return None

    if not math.isfinite(eti):
        return None

    return nf1 * (et1 / eti) ** 3.89


def fatigue_damage(repetitions, allowable_life):

    if allowable_life is None or allowable_life <= 0:
        return None

    return repetitions / allowable_life


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

    if min(
        mass,
        stiffness,
        duration,
        simulation_time
    ) <= 0:
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

    # Initial conditions
    # x(0) = 0
    # velocity(0) = 0

    displacement = np.zeros(steps)
    velocity = np.zeros(steps)
    acceleration = np.zeros(steps)

    acceleration[0] = (
        force[0]
        - damping * velocity[0]
        - stiffness * displacement[0]
    ) / mass

    # Newmark average acceleration method

    beta = 0.25
    gamma = 0.5

    a0 = 1 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = dt * (gamma / (2 * beta) - 1)

    effective_stiffness = (
        stiffness
        + a0 * mass
        + a1 * damping
    )

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
                (1 - gamma) * acceleration[i-1]
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
        minimum=0
    )

    # Shared project parameters

    mass = get_project_value("mass")
    stiffness = get_project_value("stiffness")
    load = get_project_value("vehicle_load")
    distance = get_project_value("loading_distance")
    simulation_time = get_project_value("simulation_duration")
    reference_strain = get_project_value("reference_strain")
    steps_value = get_project_value("time_steps")

    st.divider()

    st.subheader("Calculated Results")

    required_inputs = [
        storage,
        loss,
        vehicle_speed,
        repetitions,
        mass,
        stiffness,
        load,
        distance,
        simulation_time,
        reference_strain,
        steps_value
    ]

    if all(value is not None for value in required_inputs):

        if (
            storage <= 0
            or loss < 0
            or mass <= 0
            or stiffness <= 0
            or load <= 0
            or distance <= 0
            or simulation_time <= 0
            or reference_strain <= 0
            or vehicle_speed <= 0
            or repetitions < 0
        ):

            st.error("Enter valid positive project parameters.")

        else:

            steps = int(steps_value)

            if steps < 100 or steps > 100000:

                st.error(
                    "Time steps must be between 100 and 100000."
                )

            else:

                # =====================================
                # 1. DAMPING PROPERTIES
                # =====================================

                tan_delta, zeta = calculate_damping(
                    storage,
                    loss
                )

                if zeta >= 1:

                    st.error(
                        "Damping ratio must be below 1 "
                        "for this underdamped model."
                    )

                else:

                    damping_c = calculate_damping_coefficient(
                        mass,
                        stiffness,
                        zeta
                    )

                    # =====================================
                    # 2. NATURAL FREQUENCY
                    # =====================================

                    omega_n, natural_frequency = (
                        calculate_natural_frequency(
                            mass,
                            stiffness
                        )
                    )

                    omega_d = (
                        omega_n * math.sqrt(1 - zeta**2)
                    )

                    damped_frequency = (
                        omega_d / (2 * math.pi)
                    )

                    # =====================================
                    # 3. MOVING-LOAD FREQUENCY
                    # =====================================

                    (
                        speed_ms,
                        loading_duration,
                        frequency,
                        omega
                    ) = moving_load_parameters(
                        vehicle_speed,
                        distance
                    )

                    # =====================================
                    # 4. FREQUENCY RATIO AND DAF
                    # =====================================

                    daf_result = calculate_daf(
                        frequency,
                        natural_frequency,
                        zeta
                    )

                    if daf_result is None:

                        st.error(
                            "DAF calculation failed."
                        )

                    else:

                        frequency_ratio, daf = daf_result

                        # =====================================
                        # 5. STATIC DISPLACEMENT
                        # =====================================

                        static_displacement = (
                            calculate_static_displacement(
                                load,
                                stiffness
                            )
                        )

                        # =====================================
                        # 6. DAMPED DISPLACEMENT
                        # =====================================

                        damped_displacement = (
                            calculate_damped_displacement(
                                daf,
                                static_displacement
                            )
                        )

                        # =====================================
                        # 7. UNDAMPED DISPLACEMENT
                        # =====================================

                        undamped_displacement = (
                            calculate_undamped_displacement(
                                mass=mass,
                                load=load,
                                omega_n=omega_n,
                                omega=omega,
                                simulation_time=simulation_time
                            )
                        )

                        # =====================================
                        # 8. RESPONSE FACTOR
                        # =====================================

                        response_factor = (
                            calculate_response_factor(
                                damped_displacement,
                                undamped_displacement
                            )
                        )

                        if response_factor is None:

                            st.error(
                                "Response Factor cannot be calculated "
                                "because the undamped displacement is "
                                "approximately zero at the selected "
                                "simulation duration."
                            )

                        else:

                            response_reduction = (
                                1 - response_factor
                            ) * 100

                            # =====================================
                            # 9. TENSILE STRAIN
                            # =====================================

                            tensile_strain_micro = (
                                calculate_tensile_strain(
                                    response_factor,
                                    reference_strain
                                )
                            )

                            # =====================================
                            # 10. FATIGUE LIFE
                            # =====================================

                            nf = fatigue_life(
                                tensile_strain_micro
                            )

                            # =====================================
                            # 11. FATIGUE DAMAGE
                            # =====================================

                            damage = fatigue_damage(
                                repetitions,
                                nf
                            )

                            # =====================================
                            # 12. NEWMARK TRANSIENT RESPONSE
                            # =====================================

                            transient_response = solve_moving_load(
                                mass=mass,
                                stiffness=stiffness,
                                damping=damping_c,
                                load=load,
                                duration=loading_duration,
                                simulation_time=simulation_time,
                                steps=steps
                            )

                            if transient_response is not None:

                                (
                                    time,
                                    force,
                                    transient_displacement,
                                    transient_velocity,
                                    transient_acceleration
                                ) = transient_response

                                dynamic_peak = np.max(
                                    np.abs(transient_displacement)
                                )

                            else:

                                dynamic_peak = None

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
                                "### 2. Frequency Calculations"
                            )

                            a, b, c = st.columns(3)

                            a.metric(
                                "Moving-Load Frequency",
                                f"{frequency:.6f} Hz"
                            )

                            b.metric(
                                "Natural Frequency",
                                f"{natural_frequency:.6f} Hz"
                            )

                            c.metric(
                                "Frequency Ratio (r)",
                                f"{frequency_ratio:.6f}"
                            )

                            a, b, c = st.columns(3)

                            a.metric(
                                "Moving-Load Angular Frequency",
                                f"{omega:.4f} rad/s"
                            )

                            b.metric(
                                "Natural Angular Frequency",
                                f"{omega_n:.4f} rad/s"
                            )

                            c.metric(
                                "Damped Natural Frequency",
                                f"{damped_frequency:.4f} Hz"
                            )

                            # -------------------------------------

                            st.divider()

                            st.markdown(
                                "### 3. Dynamic Amplification Factor"
                            )

                            a, b, c = st.columns(3)

                            a.metric(
                                "Dynamic Peak Displacement (Newmark)",
                                f"{dynamic_peak * 1000:.6f} mm"
                                if dynamic_peak is not None
                                else "—"
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

                            st.divider()

                            st.markdown(
                                "### 4. Response Factor"
                            )

                            a, b = st.columns(2)

                            a.metric(
                                "Response Factor",
                                f"{response_factor:.6f}"
                            )

                            b.metric(
                                "Response Reduction",
                                f"{response_reduction:.4f}%"
                            )

                            # -------------------------------------

                            st.divider()

                            st.markdown(
                                "### 5. Tensile Strain Calculation"
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
                                "### 6. Fatigue Damage"
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
                                ["Vehicle Speed", vehicle_speed, "km/h"],
                                ["Vehicle Speed", speed_ms, "m/s"],
                                ["Loading Distance", distance, "m"],
                                ["Loading Duration", loading_duration, "s"],
                                ["Moving-Load Frequency", frequency, "Hz"],
                                ["Moving-Load Angular Frequency", omega, "rad/s"],
                                ["Natural Frequency", natural_frequency, "Hz"],
                                ["Natural Angular Frequency", omega_n, "rad/s"],
                                ["Damped Natural Frequency", damped_frequency, "Hz"],
                                ["Frequency Ratio", frequency_ratio, "-"],
                                ["Static Displacement", static_displacement, "m"],
                                ["Damped Displacement (DAF × Static)", damped_displacement, "m"],
                                ["Undamped Displacement at T", undamped_displacement, "m"],
                                ["Dynamic Amplification Factor", daf, "-"],
                                ["Response Factor", response_factor, "-"],
                                ["Response Reduction", response_reduction, "%"],
                                ["Reference Tensile Strain", reference_strain, "με"],
                                ["Calculated Tensile Strain", tensile_strain_micro, "με"],
                                ["Applied Load Repetitions", repetitions, "repetitions"],
                                ["Fatigue Damage", damage, "-"]
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

        st.info(
            "Enter valid material properties, vehicle speed, "
            "repetitions, and project parameters."
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

        numeric_input(
            "Equivalent Mass",
            "mass",
            "kg",
            minimum=0.000001
        )

        numeric_input(
            "Equivalent Stiffness",
            "stiffness",
            "N/m",
            minimum=0.000001
        )

        numeric_input(
            "Vehicle Load",
            "vehicle_load",
            "N",
            minimum=0.000001
        )

        numeric_input(
            "Characteristic Loading Length",
            "loading_distance",
            "m",
            minimum=0.000001
        )

    with c2:

        numeric_input(
            "Simulation Duration",
            "simulation_duration",
            "s",
            minimum=0.000001
        )

        numeric_input(
            "Temperature",
            "temperature",
            "°C"
        )

        numeric_input(
            "Reference Tensile Strain",
            "reference_strain",
            "με",
            minimum=0.000001
        )

        numeric_input(
            "Asphalt Modulus",
            "asphalt_modulus",
            "psi",
            minimum=0.000001
        )

    # Numerical time steps remain hidden.
    # Default = 4000.

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
        ["Vehicle Speed", st.session_state["vehicle_speed"], "km/h"]
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

