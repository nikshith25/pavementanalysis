"""
Flexible Pavement Response and Design Under Moving Vehicle Loads
Streamlit engineering dashboard based on the user's 2-page governing-equations PDF.

IMPORTANT:
- The reference stiffness exponent in the PDF is unreadable. K is intentionally left blank.
- Missing engineering inputs are never silently guessed.
- Fatigue strain is an independent input; DAF is NOT automatically converted to strain.
- This is a preliminary equivalent SDOF model, not a multilayer viscoelastic solver.
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp

st.set_page_config(
    page_title="Flexible Pavement Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Engineering functions -----------------------------

def loading_duration(contact_diameter_m: float, speed_m_s: float) -> float:
    if contact_diameter_m <= 0 or speed_m_s <= 0:
        raise ValueError("Contact diameter and vehicle speed must be positive.")
    return contact_diameter_m / speed_m_s


def characteristic_frequency(duration_s: float) -> float:
    if duration_s <= 0:
        raise ValueError("Loading duration must be positive.")
    return 1.0 / (2.0 * duration_s)


def equivalent_damping(
    h_bc_m: float, e_bc_pa: float, zeta_bc: float,
    h_dbm_m: float, e_dbm_pa: float, zeta_dbm: float,
) -> float:
    values = [h_bc_m, e_bc_pa, h_dbm_m, e_dbm_pa]
    if any(v <= 0 for v in values):
        raise ValueError("Layer thicknesses and elastic moduli must be positive.")
    if not (0 <= zeta_bc < 1 and 0 <= zeta_dbm < 1):
        raise ValueError("Damping ratios must be in the interval [0, 1).")
    a = h_bc_m / e_bc_pa
    b = h_dbm_m / e_dbm_pa
    return (a * zeta_bc + b * zeta_dbm) / (a + b)


def damping_coefficient(zeta_eq: float, stiffness_n_m3: float, mass_kg_m2: float) -> float:
    if not (0 <= zeta_eq < 1):
        raise ValueError("Equivalent damping ratio must be in [0, 1).")
    if stiffness_n_m3 <= 0 or mass_kg_m2 <= 0:
        raise ValueError("Stiffness and area-based mass must be positive.")
    return 2.0 * zeta_eq * math.sqrt(stiffness_n_m3 * mass_kg_m2)


def natural_frequency(stiffness_n_m3: float, mass_kg_m2: float) -> tuple[float, float]:
    if stiffness_n_m3 <= 0 or mass_kg_m2 <= 0:
        raise ValueError("Stiffness and area-based mass must be positive.")
    omega_n = math.sqrt(stiffness_n_m3 / mass_kg_m2)
    return omega_n, omega_n / (2.0 * math.pi)


def moving_load(t: float, pressure_pa: float, duration_s: float) -> float:
    if 0.0 <= t <= duration_s:
        return pressure_pa * math.sin(math.pi * t / duration_s)
    return 0.0


def simulate_mck(
    pressure_pa: float,
    duration_s: float,
    mass_kg_m2: float,
    damping_n_s_m3: float,
    stiffness_n_m3: float,
    points: int = 5000,
):
    """Solve M*x'' + C*x' + K*x = p(t), with x(0)=0 and x'(0)=0."""
    if min(pressure_pa, duration_s, mass_kg_m2, stiffness_n_m3) <= 0:
        raise ValueError("Pressure, duration, mass and stiffness must be positive.")
    if damping_n_s_m3 < 0:
        raise ValueError("Damping coefficient cannot be negative.")

    omega_n = math.sqrt(stiffness_n_m3 / mass_kg_m2)
    natural_period = 2.0 * math.pi / omega_n
    # Simulate the pulse and enough free-vibration response to capture the peak.
    end_time = max(5.0 * duration_s, 10.0 * natural_period)
    n = int(np.clip(max(points, end_time * omega_n * 100.0), points, 100000))
    times = np.linspace(0.0, end_time, n)

    def rhs(t, y):
        displacement, velocity = y
        force = moving_load(t, pressure_pa, duration_s)
        acceleration = (force - damping_n_s_m3 * velocity -
                        stiffness_n_m3 * displacement) / mass_kg_m2
        return [velocity, acceleration]

    solution = solve_ivp(
        rhs,
        (0.0, end_time),
        (0.0, 0.0),
        t_eval=times,
        method="DOP853",
        rtol=1e-8,
        atol=1e-11,
        max_step=end_time / (n - 1),
    )
    if not solution.success:
        raise RuntimeError(f"MCK numerical integration failed: {solution.message}")

    displacement = solution.y[0]
    velocity = solution.y[1]
    load = np.array([moving_load(t, pressure_pa, duration_s) for t in times])
    static_displacement = pressure_pa / stiffness_n_m3
    peak_index = int(np.argmax(np.abs(displacement)))
    daf = abs(displacement[peak_index]) / static_displacement

    return {
        "time": times,
        "displacement": displacement,
        "velocity": velocity,
        "load": load,
        "static_displacement": static_displacement,
        "peak_displacement": float(abs(displacement[peak_index])),
        "peak_time": float(times[peak_index]),
        "daf": float(daf),
        "natural_period": natural_period,
    }


def fatigue_life(strain_in_in: float, modulus_psi: float) -> float:
    if strain_in_in <= 0 or modulus_psi <= 0:
        raise ValueError("Tensile strain and modulus must be positive.")
    return 0.0796 * (1.0 / strain_in_in) ** 3.291 * (1.0 / modulus_psi) ** 0.854


def fmt(value, significant=6):
    return f"{value:.{significant}g}"


def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def chart_layout(fig, title, x_title, y_title):
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        height=420,
        margin=dict(l=25, r=25, t=65, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ----------------------------------- Sidebar -----------------------------------

st.title("Flexible Pavement Response & Design")
st.caption("Moving vehicle loads • Governing-equation calculator • Interactive visualisation")

with st.sidebar:
    st.header("Engineering inputs")
    st.caption("Reference values from the supplied PDF are prefilled. Missing values remain blank.")

    st.subheader("Tyre and vehicle")
    pressure_mpa = st.number_input("Tyre contact pressure (MPa)", min_value=0.000001,
                                   value=0.56, step=0.01, format="%.6f")
    diameter_mm = st.number_input("Tyre contact diameter (mm)", min_value=0.000001,
                                  value=213.2, step=1.0)
    speed_kmh = st.number_input("Vehicle speed (km/h)", min_value=0.0, value=None,
                                placeholder="Required for loading duration")

    st.subheader("Asphalt layers")
    h_bc_mm = st.number_input("BC thickness (mm)", min_value=0.000001, value=50.0)
    h_dbm_mm = st.number_input("DBM thickness (mm)", min_value=0.000001, value=60.0)
    zeta_bc = st.number_input("BC damping ratio", min_value=0.0, max_value=0.999999,
                              value=0.08607, format="%.5f")
    zeta_dbm = st.number_input("DBM damping ratio", min_value=0.0, max_value=0.999999,
                               value=0.19250, format="%.5f")
    e_bc_mpa = st.number_input("BC elastic modulus (MPa)", min_value=0.0, value=None,
                               placeholder="Required for equivalent damping")
    e_dbm_mpa = st.number_input("DBM elastic modulus (MPa)", min_value=0.0, value=None,
                                placeholder="Required for equivalent damping")

    st.subheader("Equivalent area-based MCK model")
    st.info("The PDF's reference K exponent is unreadable. Enter a verified K value.")
    stiffness = st.number_input("Equivalent stiffness K (N/m³)", min_value=0.0,
                                value=None, format="%.6e",
                                placeholder="Enter verified K")
    mass = st.number_input("Equivalent area mass M (kg/m²)", min_value=0.0,
                           value=None, format="%.6f",
                           placeholder="Required for C, fn and MCK")

    st.subheader("Fatigue inputs")
    strain_micro = st.number_input("Tensile strain (microstrain, με)", min_value=0.0,
                                   value=None, placeholder="Independent validated strain")
    fatigue_modulus_psi = st.number_input("Fatigue modulus E (psi)", min_value=0.0,
                                          value=None, placeholder="Required by fatigue equation")

    st.divider()
    st.caption("Unit conversions are performed internally. Do not enter MPa where psi is requested.")

# Convert to SI
p0 = pressure_mpa * 1e6
diameter = diameter_mm / 1000.0
h_bc = h_bc_mm / 1000.0
h_dbm = h_dbm_mm / 1000.0
speed = speed_kmh / 3.6 if speed_kmh is not None else None
e_bc = e_bc_mpa * 1e6 if e_bc_mpa is not None and e_bc_mpa > 0 else None
e_dbm = e_dbm_mpa * 1e6 if e_dbm_mpa is not None and e_dbm_mpa > 0 else None
strain = strain_micro * 1e-6 if strain_micro is not None and strain_micro > 0 else None

# Calculate independently where inputs are sufficient.
td = fc = zeta_eq = c_coeff = omega_n = fn = None
mck = None
fatigue_n = None
errors = []

if speed is not None and speed > 0:
    td = loading_duration(diameter, speed)
    fc = characteristic_frequency(td)

if e_bc is not None and e_dbm is not None:
    zeta_eq = equivalent_damping(h_bc, e_bc, zeta_bc, h_dbm, e_dbm, zeta_dbm)

if stiffness is not None and stiffness > 0 and mass is not None and mass > 0:
    omega_n, fn = natural_frequency(stiffness, mass)
    if zeta_eq is not None:
        c_coeff = damping_coefficient(zeta_eq, stiffness, mass)
    if td is not None:
        if zeta_eq is not None:
            c_for_sim = damping_coefficient(zeta_eq, stiffness, mass)
            try:
                mck = simulate_mck(p0, td, mass, c_for_sim, stiffness)
            except Exception as exc:
                errors.append(str(exc))
        else:
            errors.append("MCK simulation needs BC and DBM moduli to calculate the equivalent damping ratio and C.")

if strain is not None and fatigue_modulus_psi is not None and fatigue_modulus_psi > 0:
    fatigue_n = fatigue_life(strain, fatigue_modulus_psi)

# ----------------------------------- Overview -----------------------------------

st.markdown(
    """
    This dashboard implements the eight equations in the supplied project document.
    Each module calculates independently when its required inputs are available.
    """
)

with st.expander("Model assumptions and accuracy notes", expanded=False):
    st.markdown("""
    - The equivalent damping ratio is the document's compliance-weighted preliminary approximation.
    - The MCK calculation is an equivalent single-degree-of-freedom (SDOF) model.
    - M is in kg/m², K in N/m³, C in N·s/m³, pressure in Pa and displacement in metres.
    - MCK initial conditions are zero displacement and zero velocity.
    - The half-sine load pulse is applied from t = 0 to t = td; the subsequent response is free vibration.
    - Fatigue uses the document's Asphalt Institute form with strain in in/in and modulus in psi.
    - No automatic DAF-to-tensile-strain conversion is performed.
    - Layer modulus changes may require a new KENLAYER analysis and updated equivalent stiffness.
    """)

if errors:
    for err in errors:
        st.warning(err)

# ------------------------------------ Modules ------------------------------------

tabs = st.tabs([
    "1 · Loading duration",
    "2 · Loading frequency",
    "3 · Equivalent damping",
    "4 · Damping coefficient",
    "5 · Natural frequency",
    "6 · MCK response",
    "7 · DAF",
    "8 · Fatigue life",
])

# Module 1
with tabs[0]:
    st.subheader("1. Loading duration")
    st.latex(r"t_d = \frac{D}{V}")
    if td is None:
        st.info("Enter a positive vehicle speed in the sidebar to calculate loading duration.")
    else:
        a, b, c = st.columns(3)
        a.metric("Loading duration", f"{td:.8f} s")
        b.metric("Contact diameter", f"{diameter:.4f} m")
        c.metric("Vehicle speed", f"{speed:.4f} m/s")
        st.caption(f"Substitution: td = {diameter:.6f} / {speed:.6f} = {td:.8f} s")
        if st.checkbox("Visualise loading duration against vehicle speed", key="viz_td"):
            speeds = np.linspace(max(0.5, speed * 0.2), speed * 2.0, 150)
            fig = go.Figure(go.Scatter(x=speeds * 3.6, y=diameter / speeds,
                                       mode="lines", name="td"))
            fig.add_vline(x=speed * 3.6, line_dash="dash",
                          annotation_text="Selected speed")
            st.plotly_chart(chart_layout(fig, "Loading duration vs vehicle speed",
                                         "Vehicle speed (km/h)", "Loading duration (s)"),
                            use_container_width=True)

# Module 2
with tabs[1]:
    st.subheader("2. Characteristic loading frequency")
    st.latex(r"f_c = \frac{1}{2t_d}")
    if fc is None:
        st.info("Enter a positive vehicle speed to calculate characteristic loading frequency.")
    else:
        st.metric("Characteristic loading frequency", f"{fc:.6f} Hz")
        st.caption(f"Substitution: fc = 1 / (2 × {td:.8f}) = {fc:.6f} Hz")
        if st.checkbox("Visualise characteristic frequency against vehicle speed", key="viz_fc"):
            speeds = np.linspace(max(0.5, speed * 0.2), speed * 2.0, 150)
            freq = speeds / (2.0 * diameter)
            fig = go.Figure(go.Scatter(x=speeds * 3.6, y=freq, mode="lines", name="fc"))
            fig.add_vline(x=speed * 3.6, line_dash="dash",
                          annotation_text="Selected speed")
            st.plotly_chart(chart_layout(fig, "Characteristic frequency vs vehicle speed",
                                         "Vehicle speed (km/h)", "Characteristic frequency (Hz)"),
                            use_container_width=True)

# Module 3
with tabs[2]:
    st.subheader("3. Equivalent damping ratio")
    st.latex(
        r"\zeta_{eq}=\frac{(h_{BC}/E_{BC})\zeta_{BC}+(h_{DBM}/E_{DBM})\zeta_{DBM}}"
        r"{(h_{BC}/E_{BC})+(h_{DBM}/E_{DBM})}"
    )
    if zeta_eq is None:
        st.info("Enter positive BC and DBM elastic moduli in MPa.")
    else:
        st.metric("Equivalent damping ratio", f"{zeta_eq:.8f}")
        st.metric("Equivalent damping (%)", f"{100*zeta_eq:.5f}%")
        bc_weight = (h_bc / e_bc) / ((h_bc / e_bc) + (h_dbm / e_dbm))
        dbm_weight = 1.0 - bc_weight
        st.write(f"Compliance weights: BC = {bc_weight:.5f}; DBM = {dbm_weight:.5f}")
        if st.checkbox("Visualise layer damping contributions", key="viz_zeta"):
            fig = go.Figure(go.Bar(
                x=["BC contribution", "DBM contribution"],
                y=[bc_weight * zeta_bc, dbm_weight * zeta_dbm],
                text=[f"{bc_weight*zeta_bc:.5f}", f"{dbm_weight*zeta_dbm:.5f}"],
                textposition="outside",
            ))
            st.plotly_chart(chart_layout(fig, "Weighted damping contributions",
                                         "Layer", "Weighted contribution to ζeq"),
                            use_container_width=True)

# Module 4
with tabs[3]:
    st.subheader("4. Damping coefficient")
    st.latex(r"C=2\zeta_{eq}\sqrt{KM}")
    if c_coeff is None:
        st.info("Requires equivalent damping ratio, verified K (N/m³), and M (kg/m²).")
    else:
        st.metric("Damping coefficient C", f"{c_coeff:.6g} N·s/m³")
        st.caption("Calculated using the area-based MCK units specified in the PDF.")
        if st.checkbox("Visualise C against equivalent damping ratio", key="viz_c"):
            zetas = np.linspace(0.0, max(0.25, min(0.95, zeta_eq * 2.0)), 150)
            cs = 2.0 * zetas * np.sqrt(stiffness * mass)
            fig = go.Figure(go.Scatter(x=zetas, y=cs, mode="lines", name="C"))
            fig.add_vline(x=zeta_eq, line_dash="dash", annotation_text="Selected ζeq")
            st.plotly_chart(chart_layout(fig, "Damping coefficient vs damping ratio",
                                         "Equivalent damping ratio", "C (N·s/m³)"),
                            use_container_width=True)

# Module 5
with tabs[4]:
    st.subheader("5. Natural frequency")
    st.latex(r"f_n=\frac{1}{2\pi}\sqrt{\frac{K}{M}}")
    if fn is None:
        st.info("Enter positive verified K (N/m³) and M (kg/m²).")
    else:
        a, b = st.columns(2)
        a.metric("Natural frequency", f"{fn:.6f} Hz")
        b.metric("Angular natural frequency", f"{omega_n:.6f} rad/s")
        if st.checkbox("Visualise natural frequency against stiffness", key="viz_fn"):
            kvals = np.linspace(stiffness * 0.2, stiffness * 2.0, 150)
            fvals = np.sqrt(kvals / mass) / (2.0 * math.pi)
            fig = go.Figure(go.Scatter(x=kvals, y=fvals, mode="lines", name="fn"))
            fig.add_vline(x=stiffness, line_dash="dash", annotation_text="Selected K")
            st.plotly_chart(chart_layout(fig, "Natural frequency vs stiffness",
                                         "Stiffness K (N/m³)", "Natural frequency (Hz)"),
                            use_container_width=True)

# Module 6
with tabs[5]:
    st.subheader("6. MCK governing equation and moving-load pulse")
    st.latex(r"M\ddot{x}+C\dot{x}+Kx=p(t)")
    st.latex(r"p(t)=P_0\sin\left(\frac{\pi t}{t_d}\right),\quad 0\leq t\leq t_d")
    if mck is None:
        st.info("Requires vehicle speed, both layer moduli, verified K, and area-based mass M.")
    else:
        a, b, c = st.columns(3)
        a.metric("Peak absolute displacement", f"{mck['peak_displacement']:.6e} m")
        b.metric("Peak time", f"{mck['peak_time']:.6f} s")
        c.metric("Static displacement", f"{mck['static_displacement']:.6e} m")
        if st.checkbox("Visualise MCK displacement and applied load", value=True, key="viz_mck"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mck["time"], y=mck["displacement"],
                                     mode="lines", name="Displacement (m)"))
            fig.add_trace(go.Scatter(x=mck["time"], y=mck["load"] / p0,
                                     mode="lines", name="Normalised load p/P₀",
                                     yaxis="y2", line=dict(dash="dot")))
            fig.update_layout(
                title="SDOF response to the half-sine moving-load pulse",
                xaxis_title="Time (s)",
                yaxis=dict(title="Displacement (m)"),
                yaxis2=dict(title="Normalised load", overlaying="y", side="right",
                            range=[-0.05, 1.05]),
                template="plotly_white", height=480,
                margin=dict(l=25, r=25, t=65, b=35),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                "Download MCK time-history CSV",
                data=pd.DataFrame({
                    "time_s": mck["time"],
                    "displacement_m": mck["displacement"],
                    "velocity_m_s": mck["velocity"],
                    "load_pa": mck["load"],
                }).to_csv(index=False).encode("utf-8"),
                file_name="mck_time_history.csv",
                mime="text/csv",
            )

# Module 7
with tabs[6]:
    st.subheader("7. Dynamic amplification factor")
    st.latex(r"DAF=\frac{x_{dynamic,max}}{x_{static}},\qquad x_{static}=\frac{P_0}{K}")
    if mck is None:
        st.info("DAF is calculated after the MCK time-history simulation is available.")
    else:
        a, b, c = st.columns(3)
        a.metric("Dynamic amplification factor", f"{mck['daf']:.6f}")
        b.metric("Dynamic peak displacement", f"{mck['peak_displacement']:.6e} m")
        c.metric("Static displacement", f"{mck['static_displacement']:.6e} m")
        st.caption("Peak displacement is the maximum absolute displacement over the simulated time history.")
        if st.checkbox("Visualise dynamic and static displacement", key="viz_daf"):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Static", "Dynamic peak"],
                                 y=[mck["static_displacement"], mck["peak_displacement"]],
                                 text=[fmt(mck["static_displacement"]), fmt(mck["peak_displacement"])],
                                 textposition="outside"))
            st.plotly_chart(chart_layout(fig, "Static vs dynamic displacement",
                                         "Response", "Displacement (m)"),
                            use_container_width=True)

# Module 8
with tabs[7]:
    st.subheader("8. Asphalt fatigue-life estimate")
    st.latex(r"N_f=0.0796\left(\frac{1}{\epsilon_t}\right)^{3.291}"
             r"\left(\frac{1}{E}\right)^{0.854}")
    st.caption("εt must be entered as in/in (the sidebar accepts microstrain); E must be entered in psi.")
    if fatigue_n is None:
        st.info("Enter a positive, independently obtained tensile strain and fatigue modulus.")
    else:
        st.metric("Estimated fatigue life, Nf", f"{fatigue_n:.6e} repetitions")
        if st.checkbox("Visualise fatigue life against tensile strain", key="viz_fatigue"):
            strains_micro = np.logspace(
                math.log10(max(strain_micro / 10.0, 1.0)),
                math.log10(strain_micro * 10.0),
                180,
            )
            lives = np.array([
                fatigue_life(s * 1e-6, fatigue_modulus_psi) for s in strains_micro
            ])
            fig = go.Figure(go.Scatter(x=strains_micro, y=lives, mode="lines",
                                       name="Fatigue life"))
            fig.add_vline(x=strain_micro, line_dash="dash",
                          annotation_text="Selected strain")
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")
            st.plotly_chart(chart_layout(fig, "Fatigue life sensitivity to tensile strain",
                                         "Tensile strain (microstrain, με)",
                                         "Estimated fatigue repetitions, Nf"),
                            use_container_width=True)

# ------------------------------- Results export --------------------------------

st.divider()
st.subheader("Calculation status and export")

rows = [
    ("1. Loading duration", td, "s"),
    ("2. Characteristic loading frequency", fc, "Hz"),
    ("3. Equivalent damping ratio", zeta_eq, "-"),
    ("4. Damping coefficient", c_coeff, "N·s/m³"),
    ("5. Natural frequency", fn, "Hz"),
    ("6. MCK peak absolute displacement", mck["peak_displacement"] if mck else None, "m"),
    ("7. Dynamic amplification factor", mck["daf"] if mck else None, "-"),
    ("8. Asphalt fatigue life", fatigue_n, "repetitions"),
]
result_df = pd.DataFrame([
    {"Calculation": name, "Result": value if value is not None else "Not calculated — missing input",
     "Unit": unit}
    for name, value, unit in rows
])
st.dataframe(result_df, use_container_width=True, hide_index=True)
st.download_button(
    "Download calculation summary (CSV)",
    data=result_df.to_csv(index=False).encode("utf-8"),
    file_name="flexible_pavement_calculation_summary.csv",
    mime="text/csv",
)

st.warning(
    "Research-use notice: Verify all input data, unit conventions, boundary/initial conditions, "
    "and model calibration against the project methodology and an appropriate layered pavement "
    "analysis before using results for design decisions."
)
