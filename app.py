
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
    zeta = tan_delta / 2

    return tan_delta, zeta


def calculate_damping_coefficient(mass, stiffness, zeta):

    return 2 * zeta * math.sqrt(mass * stiffness)


# =========================================================
# NATURAL FREQUENCY
# =========================================================

def calculate_natural_frequency(mass, stiffness):

    omega_n = math.sqrt(stiffness / mass)

    f_n = omega_n / (2 * math.pi)

    return omega_n, f_n


# =========================================================
# MOVING VEHICLE FREQUENCY
# =========================================================

def moving_load_parameters(speed_kmh, distance):

    if speed_kmh <= 0 or distance <= 0:
        return None, None, None, None

    # Convert km/h to m/s

    speed_ms = speed_kmh / 3.6

    # Vehicle loading duration

    loading_duration = distance / speed_ms

    # USER-SPECIFIED FREQUENCY FORMULA
    # f = (v / 3.6) / D

    frequency = speed_ms / distance

    # Angular frequency
    # omega = 2*pi*f

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

    r = frequency / natural_frequency

    denominator = math.sqrt(
        (1 - r**2)**2
        + (2 * zeta * r)**2
    )

    if denominator <= 0:
        return r, None

    daf = 1 / denominator

    return r, daf


# =========================================================
# STATIC DISPLACEMENT
# =========================================================

def calculate_static_displacement(load, stiffness):

    return load / stiffness


# =========================================================
# UNDAMPED DISPLACEMENT HISTORY
# =========================================================

def calculate_undamped_history(
    mass,
    load,
    omega_n,
    omega,
    simulation_time,
    steps
):

    """
    User-specified equation:

    x_u(t) = (P0/M)/(wn²-w²)
             * [sin(wt) - (w/wn)*sin(wn*t)]

    Initial conditions:
    x(0) = 0
    velocity(0) = 0
    """

    time = np.linspace(
        0,
        simulation_time,
        steps
    )

    denominator = omega_n**2 - omega**2

    tolerance = (
        1e-10
        * max(omega_n**2, omega**2, 1.0)
    )

    if abs(denominator) <= tolerance:

        # Resonance limit

        displacement = (
            (load / mass)
            / (2 * omega_n**2)
            * (
                np.sin(omega_n * time)
                - omega_n * time * np.cos(omega_n * time)
            )
        )

    else:

        displacement = (
            (load / mass) / denominator
            * (
                np.sin(omega * time)
                - (omega / omega_n)
                * np.sin(omega_n * time)
            )
        )

    return time, displacement


# =========================================================
# DAMPED DISPLACEMENT HISTORY
# NEWMARK-BETA METHOD
# =========================================================

def calculate_damped_history(
    mass,
    stiffness,
    damping,
    load,
    omega,
    simulation_time,
    steps
):

    """
    Equation of motion:

    M*x'' + C*x' + K*x = P0*sin(omega*t)

    Initial conditions:

    x(0) = 0
    velocity(0) = 0

    Newmark average acceleration method:
    beta = 0.25
    gamma = 0.5
    """

    time = np.linspace(
        0,
        simulation_time,
        steps
    )

    dt = time[1] - time[0]

    # Same sinusoidal loading used for the
    # undamped displacement equation.

    force = load * np.sin(omega * time)

    # Newmark parameters

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

    displacement = np.zeros(steps)
    velocity = np.zeros(steps)
    acceleration = np.zeros(steps)

    # Initial conditions

    displacement[0] = 0
    velocity[0] = 0

    acceleration[0] = (
        force[0]
        - damping * velocity[0]
        - stiffness * displacement[0]
    ) / mass

    # Time integration

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
# RESPONSE FACTOR
# =========================================================

def calculate_response_factor(
    damped_maximum,
    undamped_maximum
):

    if undamped_maximum <= 1e-15:
        return None

    return damped_maximum / undamped_maximum


# =========================================================
# TENSILE STRAIN
# =========================================================

def calculate_tensile_strain(
    response_factor,
    reference_strain
):

    # Both values are in microstrain.

    return response_factor * reference_strain


# =========================================================
# FATIGUE LIFE
# =========================================================

def fatigue_life(calculated_tensile_strain):

    # Nfi = 98.58 * (112.2 / eti)^3.89

    nf1 = 98.58
    et1 = 112.2

    if calculated_tensile_strain <= 0:
        return None

    return nf1 * (
        et1 / calculated_tensile_strain
    ) ** 3.89


# =========================================================
# FATIGUE DAMAGE
# =========================================================

def fatigue_damage(repetitions, allowable_life):

    if allowable_life is None or allowable_life <= 0:
        return None

    return repetitions / allowable_life


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

    # Project parameters

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
            or vehicle_speed <= 0
            or repetitions < 0
            or mass <= 0
            or stiffness <= 0
            or load <= 0
            or distance <= 0
            or simulation_time <= 0
            or reference_strain <= 0
        ):

            st.error(
                "Enter valid project parameters."
            )

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

                    frequency_ratio, daf = calculate_daf(
                        frequency,
                        natural_frequency,
                        zeta
                    )

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
                    # 6. DAMPED DISPLACEMENT HISTORY
                    # =====================================

                    (
                        time_damped,
                        force,
                        damped_history,
                        damped_velocity,
                        damped_acceleration
                    ) = calculate_damped_history(
                        mass=mass,
                        stiffness=stiffness,
                        damping=damping_c,
                        load=load,
                        omega=omega,
                        simulation_time=simulation_time,
                        steps=steps
                    )

                    # =====================================
                    # 7. UNDAMPED DISPLACEMENT HISTORY
                    # =====================================

                    (
                        time_undamped,
                        undamped_history
                    ) = calculate_undamped_history(
                        mass=mass,
                        load=load,
                        omega_n=omega_n,
                        omega=omega,
                        simulation_time=simulation_time,
                        steps=steps
                    )

                    # =====================================
                    # 8. MAXIMUM DISPLACEMENTS
                    # =====================================

                    damped_index = np.argmax(
                        np.abs(damped_history)
                    )

                    undamped_index = np.argmax(
                        np.abs(undamped_history)
                    )

                    damped_maximum = float(
                        np.abs(damped_history[damped_index])
                    )

                    undamped_maximum = float(
                        np.abs(undamped_history[undamped_index])
                    )

                    damped_peak_time = float(
                        time_damped[damped_index]
                    )

                    undamped_peak_time = float(
                        time_undamped[undamped_index]
                    )

                    # =====================================
                    # 9. RESPONSE FACTOR
                    # =====================================

                    response_factor = calculate_response_factor(
                        damped_maximum,
                        undamped_maximum
                    )

                    if response_factor is None:

                        st.error(
                            "Response Factor cannot be calculated "
                            "because undamped maximum displacement "
                            "is approximately zero."
                        )

                    else:

                        response_reduction = (
                            1 - response_factor
                        ) * 100

                        # =====================================
                        # 10. TENSILE STRAIN
                        # =====================================

                        tensile_strain_micro = (
                            calculate_tensile_strain(
                                response_factor,
                                reference_strain
                            )
                        )

                        # =====================================
                        # 11. FATIGUE LIFE
                        # =====================================

                        nf = fatigue_life(
                            tensile_strain_micro
                        )

                        # =====================================
                        # 12. FATIGUE DAMAGE
                        # =====================================

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
                            "Frequency Ratio",
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
                            "Static Displacement",
                            f"{static_displacement * 1000:.6f} mm"
                        )

                        a = a

                        b.metric(
                            "Dynamic Amplification Factor",
                            f"{daf:.6f}" if daf is not None else "—"
                        )

                        c.metric(
                            "DAF-Based Displacement",
                            f"{daf * static_displacement * 1000:.6f} mm"
                            if daf is not None
                            else "—"
                        )

                        # -------------------------------------

                        st.divider()

                        st.markdown(
                            "### 4. Transient Displacement Comparison"
                        )

                        a, b = st.columns(2)

                        a.metric(
                            "Maximum Damped Displacement",
                            f"{damped_maximum * 1000:.6f} mm"
                        )

                        b.metric(
                            "Maximum Undamped Displacement",
                            f"{undamped_maximum * 1000:.6f} mm"
                        )

                        a, b = st.columns(2)

                        a.metric(
                            "Damped Peak Time",
                            f"{damped_peak_time:.6f} s"
                        )

                        b.metric(
                            "Undamped Peak Time",
                            f"{undamped_peak_time:.6f} s"
                        )

                        # -------------------------------------

                        st.divider()

                        st.markdown(
                            "### 5. Response Factor"
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
                            "### 6. Tensile Strain Calculation"
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
                            "### 7. Fatigue Damage"
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
                            ["Simulation Duration", simulation_time, "s"],
                            ["Moving-Load Frequency", frequency, "Hz"],
                            ["Moving-Load Angular Frequency", omega, "rad/s"],
                            ["Natural Frequency", natural_frequency, "Hz"],
                            ["Natural Angular Frequency", omega_n, "rad/s"],
                            ["Damped Natural Frequency", damped_frequency, "Hz"],
                            ["Frequency Ratio", frequency_ratio, "-"],
                            ["Static Displacement", static_displacement, "m"],
                            ["DAF", daf, "-"],
                            [
                                "DAF-Based Displacement",
                                daf * static_displacement,
                                "m"
                            ],
                            [
                                "Maximum Damped Displacement",
                                damped_maximum,
                                "m"
                            ],
                            [
                                "Maximum Undamped Displacement",
                                undamped_maximum,
                                "m"
                            ],
                            [
                                "Damped Peak Time",
                                damped_peak_time,
                                "s"
                            ],
                            [
                                "Undamped Peak Time",
                                undamped_peak_time,
                                "s"
                            ],
                            ["Response Factor", response_factor, "-"],
                            ["Response Reduction", response_reduction, "%"],
                            ["Reference Tensile Strain", reference_strain, "με"],
                            ["Calculated Tensile Strain", tensile_strain_micro, "με"],
                            ["Allowable Fatigue Life", nf, "repetitions"],
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

