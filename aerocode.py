import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize_scalar

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SECTION 1: CORE AERODYNAMIC & THERMODYNAMIC FUNCTIONS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def calculate_isentropic_ratios(Ma, k):
    """Calculates all isentropic ratios from a given Mach number."""
    results = {}
    if Ma is None or np.isnan(Ma): return None
    results['Ma'] = Ma
    t0_t_term = 1 + (k - 1) / 2 * Ma**2
    results['T0/T'] = t0_t_term
    results['p0/p'] = t0_t_term**(k / (k - 1))
    results['rho0/rho'] = t0_t_term**(1 / (k - 1))
    if Ma == 0:
        results['A/A*'] = np.inf
    else:
        term1 = 2 / (k + 1)
        term2 = 1 + (k - 1) / 2 * Ma**2
        results['A/A*'] = (1 / Ma) * ((term1 * term2)**((k + 1) / (2 * (k - 1))))
    return results

def solve_for_ma(ratio_val, ratio_type, k, flow_regime):
    """Solves for Mach number given a ratio value using a root finder."""
    if ratio_type == "p0/p":
        eqn = lambda m: (1 + (k - 1) / 2 * m**2)**(k / (k - 1)) - ratio_val
    elif ratio_type == "T0/T":
        eqn = lambda m: (1 + (k - 1) / 2 * m**2) - ratio_val
    elif ratio_type == "rho0/rho":
        eqn = lambda m: (1 + (k - 1) / 2 * m**2)**(1 / (k - 1)) - ratio_val
    elif ratio_type == "A/A*":
        term1 = 2 / (k + 1)
        eqn = lambda m: (1/m) * ((term1 * (1 + (k-1)/2 * m**2))**((k+1)/(2*(k-1)))) - ratio_val
    else: return np.nan
    bracket = [0.0001, 1.0] if flow_regime == 'subsonic' else [1.0001, 20.0]
    try:
        sol = root_scalar(eqn, bracket=bracket)
        return sol.root
    except ValueError: return np.nan

def calculate_normal_shock_relations(M1, k):
    """Calculates post-shock conditions from pre-shock Mach number M1."""
    results = {}; M1_sq = M1**2; results['M1'] = M1
    num = M1_sq + 2 / (k - 1)
    den = (2 * k / (k - 1)) * M1_sq - 1
    results['M2'] = np.sqrt(num / den)
    results['p2/p1'] = 1 + (2 * k / (k + 1)) * (M1_sq - 1)
    results['rho2/rho1'] = ((k + 1) * M1_sq) / ((k - 1) * M1_sq + 2)
    results['T2/T1'] = results['p2/p1'] / results['rho2/rho1']
    p0_ratio_term_M1 = (1 + (k - 1) / 2 * M1_sq)
    p0_ratio_term_M2 = (1 + (k - 1) / 2 * results['M2']**2)
    results['p02/p01'] = results['p2/p1'] * (p0_ratio_term_M2 / p0_ratio_term_M1)**(k / (k - 1))
    return results

def solve_rayleigh_pitot(p02_p1_ratio, k):
    """Solves Rayleigh Pitot formula for supersonic Mach number."""
    eqn = lambda m: ((((k + 1)**2 * m**2) / (4 * k * m**2 - 2 * (k - 1)))**(k / (k - 1)) * ((1 - k + 2 * k * m**2) / (k + 1)) - p02_p1_ratio)
    try:
        sol = root_scalar(eqn, bracket=[1.0001, 20.0])
        return sol.root
    except ValueError: return np.nan

def calculate_fanno_ratios(Ma, k):
    """Calculates all Fanno flow ratios from a given Mach number."""
    if Ma is None or np.isnan(Ma) or Ma == 0: return None
    results = {}; Ma_sq = Ma**2
    results['T/T*'] = (k + 1) / (2 + (k - 1) * Ma_sq)
    results['p/p*'] = (1 / Ma) * np.sqrt((k + 1) / (2 + (k - 1) * Ma_sq))
    results['p0/p0*'] = (1 / Ma) * (((2 + (k - 1) * Ma_sq) / (k + 1))**((k + 1) / (2 * (k - 1))))
    term1 = (1 - Ma_sq) / (k * Ma_sq)
    term2 = ((k + 1) / (2 * k)) * np.log(((k + 1) * Ma_sq) / (2 + (k - 1) * Ma_sq))
    results['fL*/D'] = term1 + term2
    return results

def solve_ma_from_fanno_length(fL_D_val, k, flow_regime):
    """Solves for Mach number given the fL*/D value."""
    eqn = lambda m: (((1 - m**2) / (k * m**2)) + (((k + 1) / (2 * k)) * np.log(((k + 1) * m**2) / (2 + (k - 1) * m**2)))) - fL_D_val
    bracket = [0.0001, 1.0] if flow_regime == 'subsonic' else [1.0001, 20.0]
    try:
        sol = root_scalar(eqn, bracket=bracket)
        return sol.root
    except (ValueError, RuntimeError): return np.nan

# --- NEW FUNCTIONS FOR WEEK 5/6 ---

def get_theta_from_beta(beta_deg, Ma1, k):
    """Calculates deflection angle theta from beta and Ma1 using TBM relation."""
    if Ma1 <= 1.0: return 0
    beta_rad = np.radians(beta_deg)
    Ma1_sq = Ma1**2
    sin_beta_sq = np.sin(beta_rad)**2
    cot_beta = 1 / np.tan(beta_rad)
    cos_2beta = np.cos(2 * beta_rad)
    
    numerator = Ma1_sq * sin_beta_sq - 1
    denominator = Ma1_sq * (k + cos_2beta) + 2
    
    if denominator == 0: return 0
    tan_theta = 2 * cot_beta * (numerator / denominator)
    return np.degrees(np.arctan(tan_theta))

def get_oblique_shock_properties(Ma1, beta_deg, k):
    """Calculates all properties for an oblique shock given Ma1 and beta."""
    if Ma1 <= 1.0: return None
    
    beta_rad = np.radians(beta_deg)
    Ma1_n = Ma1 * np.sin(beta_rad)
    
    if Ma1_n <= 1.0: # Not a shock (Mach wave or invalid)
        return None 
        
    norm_shock_res = calculate_normal_shock_relations(Ma1_n, k)
    theta_deg = get_theta_from_beta(beta_deg, Ma1, k)
    theta_rad = np.radians(theta_deg)
    
    Ma2_n = norm_shock_res['M2']
    Ma2 = Ma2_n / np.sin(beta_rad - theta_rad)
    
    results = {
        'Ma1': Ma1, 'β (deg)': beta_deg, 'θ (deg)': theta_deg,
        'Ma1_n': Ma1_n, 'Ma2_n': Ma2_n, 'Ma2': Ma2,
        'p2/p1': norm_shock_res['p2/p1'],
        'T2/T1': norm_shock_res['T2/T1'],
        'rho2/rho1': norm_shock_res['rho2/rho1'],
        'p02/p01': norm_shock_res['p02/p01']
    }
    return results

def get_max_theta(Ma1, k):
    """Finds the maximum deflection angle (theta_max) for a given Ma1."""
    if Ma1 <= 1.0: return (0, 90)
    mu_rad = np.arcsin(1 / Ma1)
    mu_deg = np.degrees(mu_rad)
    
    # Find the beta that gives the maximum theta
    opt_result = minimize_scalar(
        lambda b: -get_theta_from_beta(b, Ma1, k), 
        bounds=(mu_deg, 90), 
        method='bounded'
    )
    
    if opt_result.success:
        theta_max = -opt_result.fun
        beta_at_max = opt_result.x
        return (theta_max, beta_at_max)
    else:
        return (0, 90) # Failed

def solve_beta(Ma1, theta, k):
    """Finds weak and strong beta solutions for a given Ma1 and theta."""
    mu_deg = np.degrees(np.arcsin(1/Ma1))
    theta_max, beta_at_max = get_max_theta(Ma1, k)
    
    if theta > theta_max:
        return (np.nan, np.nan, theta_max) # Detached shock
    
    eqn = lambda b: get_theta_from_beta(b, Ma1, k) - theta
    
    try:
        beta_weak = root_scalar(eqn, bracket=[mu_deg + 0.01, beta_at_max - 0.01]).root
    except ValueError:
        beta_weak = np.nan
        
    try:
        beta_strong = root_scalar(eqn, bracket=[beta_at_max + 0.01, 89.99]).root
    except ValueError:
        beta_strong = np.nan
        
    return (beta_weak, beta_strong, theta_max)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SECTION 2: STREAMLIT USER INTERFACE
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Conversion constants
PA_TO_ATM = 1 / 101325.0
K_TO_C = -273.15

st.set_page_config(layout="wide")
st.title("Aerodynamics Toolbox")
st.markdown("A tool for solving common compressible flow and aerodynamics problems based on lecture notes.")

# --- UNIT CONVERTER (MOVED TO TOP) ---
with st.expander("**Unit Converters**", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Temperature")
        temp_val = st.number_input("Temperature Value", key="temp_conv_val", value=25.0)
        temp_unit = st.radio("Convert from:", ('°C to K', 'K to °C'), key="temp_conv_unit")
        if temp_unit == '°C to K':
            st.metric("Result", f"{temp_val - K_TO_C:.2f} K")
        else:
            st.metric("Result", f"{temp_val + K_TO_C:.2f} °C")
    with col2:
        st.subheader("Pressure")
        pres_val = st.number_input("Pressure Value", key="pres_conv_val", value=1.0)
        pres_unit = st.radio("Convert from:", ('atm to Pa', 'Pa to atm'), key="pres_conv_unit")
        if pres_unit == 'atm to Pa':
            st.metric("Result", f"{pres_val / PA_TO_ATM:.1f} Pa")
        else:
            st.metric("Result", f"{pres_val * PA_TO_ATM:.6f} atm")

# Create tabs for different functionalities
tab_basic, tab_week1, tab_week2, tab_week3, tab_week4, tab_week5, tab_week6 = st.tabs([
    "Basic Calculators",
    "Week 1: Aeroplane Drag",
    "Week 2: Isentropic Nozzle Flow",
    "Week 3: Fanno Flow & Wind Tunnels",
    "Week 4: Shocks & Measurements",
    "Week 5: Oblique Shocks",
    "Week 6: θ-β-Ma & Reflections"
])

# --- BASIC CALCULATORS TAB ---
with tab_basic:
    st.header("Fundamental Equation Solvers")
    st.markdown("Solve for individual variables from core equations.")

    # --- Ideal Gas Law ---
    with st.expander("**Ideal Gas Law (Session 1b)**: `p = ρ * R * T`"):
        col1, col2 = st.columns(2)
        with col1:
            st.latex("p = \\rho R T")
            solve_for_ig = st.radio("Solve for:", ('p', 'ρ', 'T'), key="ig_solve")
            p_ig = st.number_input("p (Pressure, Pa)", value=101325.0, format="%.1f", key="p_ig", disabled=(solve_for_ig=='p'))
            rho_ig = st.number_input("ρ (Density, kg/m³)", value=1.225, format="%.4f", key="rho_ig", disabled=(solve_for_ig=='ρ'))
            R_ig = st.number_input("R (Gas Constant, J/kg·K)", value=287.0, format="%.1f", key="R_ig")
            T_ig = st.number_input("T (Temperature, K)", value=288.15, format="%.2f", key="T_ig", disabled=(solve_for_ig=='T'))
        with col2:
            st.subheader("Result")
            try:
                if solve_for_ig == 'p': 
                    result = rho_ig * R_ig * T_ig
                    st.metric("Pressure (p)", f"{result:.1f} Pa | {result * PA_TO_ATM:.4f} atm")
                elif solve_for_ig == 'ρ': 
                    result = p_ig / (R_ig * T_ig)
                    st.metric("Density (ρ)", f"{result:.4f} kg/m³")
                elif solve_for_ig == 'T': 
                    result = p_ig / (rho_ig * R_ig)
                    st.metric("Temperature (T)", f"{result:.2f} K | {result + K_TO_C:.2f} °C")
            except ZeroDivisionError: st.error("Input values cannot be zero.")

    # --- Speed of Sound Calculator ---
    with st.expander("**Speed of Sound (Session 1b)**: `c = sqrt(k * R * T)`"):
        col1, col2 = st.columns(2)
        with col1:
            st.latex("c = \sqrt{k R T}")
            k_c = st.number_input("k", value=1.4, format="%.3f", key="k_c")
            R_c = st.number_input("R (J/kg·K)", value=287.0, format="%.1f", key="R_c")
            T_c = st.number_input("T (K)", value=288.15, format="%.2f", key="T_c")
        with col2:
            st.subheader("Result")
            if T_c < 0: st.error("Temperature must be in Kelvin (>= 0).")
            else: c_val = np.sqrt(k_c * R_c * T_c); st.metric("Speed of Sound (c)", f"{c_val:.2f} m/s")

    # --- Mach Number Definition ---
    with st.expander("**Mach Number Definition (Session 1b)**: `Ma = u / c`"):
        col1, col2 = st.columns(2)
        with col1:
            st.latex("Ma = u / c")
            solve_for_ma_def = st.radio("Solve for:", ('Ma', 'u', 'c'), key="ma_def_solve")
            Ma_def = st.number_input("Ma", value=0.8, format="%.4f", key="Ma_def", disabled=(solve_for_ma_def=='Ma'))
            u_def = st.number_input("u (m/s)", value=272.0, format="%.2f", key="u_def", disabled=(solve_for_ma_def=='u'))
            c_def = st.number_input("c (m/s)", value=340.0, format="%.2f", key="c_def", disabled=(solve_for_ma_def=='c'))
        with col2:
            st.subheader("Result")
            try:
                if solve_for_ma_def == 'Ma': result = u_def / c_def; st.metric("Mach Number (Ma)", f"{result:.4f}")
                elif solve_for_ma_def == 'u': result = Ma_def * c_def; st.metric("Velocity (u)", f"{result:.2f} m/s")
                elif solve_for_ma_def == 'c': result = u_def / Ma_def; st.metric("Speed of Sound (c)", f"{result:.2f} m/s")
            except ZeroDivisionError: st.error("Denominator cannot be zero.")

    # --- Isentropic Stagnation Relations ---
    with st.expander("**Isentropic Stagnation Relations (Session 1b/2a)**"):
        st.latex("T_0 = T \\left(1 + \\frac{k-1}{2} Ma^2\\right) \quad | \quad p_0 = p \\left(1 + \\frac{k-1}{2} Ma^2\\right)^{\\frac{k}{k-1}}")
        solve_for_isen = st.radio("Solve for:", ('Stagnation (T0, p0)', 'Static (T, p)', 'Mach Number'), key="isen_solve")
        col1, col2, col3 = st.columns(3); k_isen_ind = col1.number_input("k", value=1.4, format="%.3f", key="k_isen_ind")
        Ma_isen_ind = col1.number_input("Mach Number", value=0.8, format="%.4f", key="Ma_isen_ind", disabled=(solve_for_isen=='Mach Number'))
        T_isen_ind = col2.number_input("Static Temp (T), K", value=288.0, format="%.2f", key="T_isen_ind", disabled=(solve_for_isen=='Static (T, p)'))
        p_isen_ind = col2.number_input("Static Pressure (p), Pa", value=101325.0, format="%.1f", key="p_isen_ind", disabled=(solve_for_isen=='Static (T, p)'))
        T0_isen_ind = col3.number_input("Stagnation Temp (T0), K", value=325.15, format="%.2f", key="T0_isen_ind", disabled=(solve_for_isen=='Stagnation (T0, p0)'))
        p0_isen_ind = col3.number_input("Stagnation Pressure (p0), Pa", value=152421.0, format="%.1f", key="p0_isen_ind", disabled=(solve_for_isen=='Stagnation (T0, p0)'))
        st.subheader("Result")
        try:
            if solve_for_isen == 'Stagnation (T0, p0)':
                t_ratio = 1 + (k_isen_ind - 1) / 2 * Ma_isen_ind**2; res_T0 = T_isen_ind * t_ratio; res_p0 = p_isen_ind * t_ratio**(k_isen_ind / (k_isen_ind - 1))
                st.metric("Stagnation Temperature (T0)", f"{res_T0:.2f} K | {res_T0 + K_TO_C:.2f} °C")
                st.metric("Stagnation Pressure (p0)", f"{res_p0:.1f} Pa | {res_p0 * PA_TO_ATM:.4f} atm")
            elif solve_for_isen == 'Static (T, p)':
                t_ratio = 1 + (k_isen_ind - 1) / 2 * Ma_isen_ind**2; res_T = T0_isen_ind / t_ratio; res_p = p0_isen_ind / t_ratio**(k_isen_ind / (k_isen_ind - 1))
                st.metric("Static Temperature (T)", f"{res_T:.2f} K | {res_T + K_TO_C:.2f} °C")
                st.metric("Static Pressure (p)", f"{res_p:.1f} Pa | {res_p * PA_TO_ATM:.4f} atm")
            elif solve_for_isen == 'Mach Number':
                t_ratio_val = T0_isen_ind / T_isen_ind; val_inside_sqrt = (t_ratio_val - 1) * 2 / (k_isen_ind - 1)
                if val_inside_sqrt < 0: st.error("Invalid temperature inputs. T0 must be >= T.")
                else: res_Ma = np.sqrt(val_inside_sqrt); st.metric("Mach Number (Ma)", f"{res_Ma:.4f}")
        except Exception as e: st.error(f"Calculation failed. Check inputs. Error: {e}")

    # --- Critical Properties Calculator ---
    with st.expander("**Critical Properties (Session 1b)**: `T*`, `p*`, and `ρ*` at Ma=1"):
        col1, col2 = st.columns(2)
        with col1:
            st.latex("""
            T^* = T_0 \\left(\\frac{2}{k+1}\\right) \quad | \quad 
            p^* = p_0 \\left(\\frac{2}{k+1}\\right)^{\\frac{k}{k-1}} \quad | \quad
            \\rho^* = \\rho_0 \\left(\\frac{2}{k+1}\\right)^{\\frac{1}{k-1}}
            """)
            k_crit = st.number_input("k", value=1.4, format="%.3f", key="k_crit")
            R_crit = st.number_input("R (J/kg·K)", value=287.0, format="%.1f", key="R_crit")
            T0_crit = st.number_input("T0 (K)", value=300.0, format="%.2f", key="T0_crit")
            p0_crit = st.number_input("p0 (Pa)", value=314000.0, format="%.1f", key="p0_crit")
        with col2:
            st.subheader("Results")
            try:
                rho0_crit = p0_crit / (R_crit * T0_crit)
                T_star = T0_crit * (2 / (k_crit + 1))
                p_star = p0_crit * (2 / (k_crit + 1))**(k_crit / (k_crit - 1))
                rho_star = rho0_crit * (2 / (k_crit + 1))**(1 / (k_crit - 1))
                st.metric("Critical Temperature (T*)", f"{T_star:.2f} K | {T_star + K_TO_C:.2f} °C")
                st.metric("Critical Pressure (p*)", f"{p_star:.1f} Pa | {p_star * PA_TO_ATM:.4f} atm")
                st.metric("Critical Density (ρ*)", f"{rho_star:.4f} kg/m³")
            except ZeroDivisionError: st.error("R and T0 must not be zero.")


    # --- Mass Flow Rate Calculator ---
    with st.expander("**Mass Flow Rate (Session 2a)**: `ṁ = ρ * u * A`"):
        col1, col2 = st.columns(2)
        with col1:
            st.latex("\dot{m} = \\rho u A")
            rho_m = st.number_input("ρ (kg/m³)", value=1.225, format="%.4f", key="rho_m")
            u_m = st.number_input("u (m/s)", value=100.0, format="%.2f", key="u_m")
            A_m = st.number_input("A (m²)", value=0.1, format="%.4f", key="A_m")
        with col2:
            st.subheader("Result"); m_dot = rho_m * u_m * A_m; st.metric("Mass Flow Rate (ṁ)", f"{m_dot:.4f} kg/s")

    # --- Choked Mass Flow Rate ---
    with st.expander("**Choked Mass Flow Rate (Session 2a)**"):
        st.latex("\\dot{m}_{max} = A^* p_0 \\sqrt{\\frac{k}{R T_0}} \\left( \\frac{2}{k+1} \\right)^{\\frac{k+1}{2(k-1)}}")
        col1, col2 = st.columns(2)
        with col1:
            A_star_choked = st.number_input("A* (m²)", value=0.0020, format="%.6f", key="A_star_choked")
            p0_choked = st.number_input("p0 (Pa)", value=1.0E6, format="%.1f", key="p0_choked")
            T0_choked = st.number_input("T0 (K)", value=800.0, format="%.2f", key="T0_choked")
            k_choked = st.number_input("k", value=1.4, format="%.3f", key="k_choked")
            R_choked = st.number_input("R (J/kg·K)", value=287.0, format="%.1f", key="R_choked")
        with col2:
            st.subheader("Result")
            try:
                term1 = A_star_choked * p0_choked * np.sqrt(k_choked / (R_choked * T0_choked))
                term2 = (2 / (k_choked + 1))**((k_choked + 1) / (2 * (k_choked - 1)))
                m_dot_max = term1 * term2
                st.metric("Max Mass Flow (ṁ_max)", f"{m_dot_max:.4f} kg/s")
            except ZeroDivisionError: st.error("Inputs cannot be zero.")

    # --- Normal Shock Static Pressure Ratio ---
    with st.expander("**Normal Shock Static Pressure Ratio (Session 4a)**"):
        st.latex("\\frac{p_2}{p_1} = 1 + \\frac{2k}{k+1} (M_1^2 - 1)")
        col1, col2 = st.columns(2)
        with col1:
            solve_for_ns = st.radio("Solve for:", ('p2/p1', 'M1'), key="ns_solve")
            M1_ns = st.number_input("M1", value=2.0, format="%.4f", key="M1_ns", disabled=(solve_for_ns=='M1'))
            p_ratio_ns = st.number_input("p2/p1", value=4.5, format="%.4f", key="p_ratio_ns", disabled=(solve_for_ns=='p2/p1'))
            k_ns = st.number_input("k", value=1.4, format="%.3f", key="k_ns")
        with col2:
            st.subheader("Result")
            try:
                if solve_for_ns == 'p2/p1':
                    result = 1 + (2 * k_ns / (k_ns + 1)) * (M1_ns**2 - 1)
                    st.metric("Pressure Ratio (p2/p1)", f"{result:.4f}")
                elif solve_for_ns == 'M1':
                    if p_ratio_ns <= 1.0: st.error("p2/p1 must be > 1 for a shock.")
                    else:
                        val_inside_sqrt = (p_ratio_ns - 1) * ((k_ns + 1) / (2 * k_ns)) + 1
                        result = np.sqrt(val_inside_sqrt)
                        st.metric("Upstream Mach Number (M1)", f"{result:.4f}")
            except Exception as e: st.error(f"Calculation failed: {e}")

    # --- Drag Polar Equation Calculator ---
    with st.expander("**Drag Polar Equation (Session 1c)**: `CD = CD0 + CL² / (π * e * AR)`"):
        col1, col2 = st.columns(2)
        with col1:
            st.latex("C_D = C_{D0} + \\frac{C_L^2}{\pi e AR}")
            solve_for = st.radio("Solve for:", ('CD', 'CL', 'CD0'), key="drag_solve")
            CL_d = st.number_input("CL", value=0.5, format="%.4f", key="CL_d", disabled=(solve_for=='CL'))
            CD_d = st.number_input("CD", value=0.035, format="%.4f", key="CD_d", disabled=(solve_for=='CD'))
            CDo_d = st.number_input("CD0", value=0.02, format="%.4f", key="CDo_d", disabled=(solve_for=='CD0'))
            e_d = st.number_input("e", value=0.8, format="%.3f", key="e_d")
            AR_d = st.number_input("AR", value=7.5, format="%.2f", key="AR_d")
        with col2:
            st.subheader("Result")
            try:
                if solve_for == 'CD':
                    result = CDo_d + (CL_d**2) / (np.pi * e_d * AR_d); st.metric("Drag Coefficient (CD)", f"{result:.4f}")
                elif solve_for == 'CL':
                    val_inside_sqrt = (CD_d - CDo_d) * (np.pi * e_d * AR_d)
                    if val_inside_sqrt < 0: st.error("Invalid inputs. CD must be > CD0.")
                    else: result = np.sqrt(val_inside_sqrt); st.metric("Lift Coefficient (CL)", f"{result:.4f}")
                elif solve_for == 'CD0':
                    result = CD_d - (CL_d**2) / (np.pi * e_d * AR_d); st.metric("Zero-Lift Drag (CD0)", f"{result:.4f}")
            except Exception as e: st.error(f"An error occurred: {e}")
    
    # --- Max L/D Condition ---
    with st.expander("**Max L/D Condition (Session 1c)**"):
        st.latex("C_{L,max(L/D)} = \sqrt{C_{D0} \pi e AR} \quad | \quad (L/D)_{max} = \\frac{\sqrt{\pi e AR / C_{D0}}}{2}")
        col1, col2 = st.columns(2)
        with col1:
            CDo_ld = st.number_input("CD0", value=0.025, format="%.4f", key="CDo_ld")
            e_ld = st.number_input("e", value=0.85, format="%.3f", key="e_ld")
            AR_ld = st.number_input("AR", value=8.0, format="%.2f", key="AR_ld")
        with col2:
            st.subheader("Results")
            try:
                CL_maxld = np.sqrt(CDo_ld * np.pi * e_ld * AR_ld)
                LD_max = np.sqrt(np.pi * e_ld * AR_ld / CDo_ld) / 2
                st.metric("CL for Max L/D", f"{CL_maxld:.4f}")
                st.metric("Max L/D Ratio", f"{LD_max:.2f}")
            except (ValueError, ZeroDivisionError): st.error("Check inputs (CD0 must be > 0).")


# --- WEEK 1: AEROPLANE DRAG TAB ---
with tab_week1:
    st.header("Aeroplane Drag Polar (Session 1c)")
    st.latex("C_D = C_{D0} + \\frac{C_L^2}{\pi e AR}")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Aircraft Parameters")
        CDo = st.number_input("Zero-Lift Drag Coefficient (CD,0)", value=0.025, format="%.4f", key="cdo_w1")
        AR = st.number_input("Aspect Ratio (AR)", value=8.0, format="%.2f", key="ar_w1")
        e = st.number_input("Oswald Efficiency Factor (e)", value=0.85, min_value=0.0, max_value=1.0, format="%.3f", key="e_w1")
        
        st.subheader("Operating Point")
        CL_point = st.slider("Lift Coefficient (CL) for analysis", 0.0, 1.5, 0.5, key="cl_slider_w1")
        
        st.subheader("Flight Conditions (for absolute values)")
        rho_w1 = st.number_input("Air Density (ρ), kg/m³", value=1.225, format="%.4f", key="rho_w1")
        V_w1 = st.number_input("Velocity (V), m/s", value=100.0, format="%.2f", key="V_w1")
        S_w1 = st.number_input("Wing Area (S), m²", value=16.16, format="%.2f", key="S_w1")

    with col2:
        st.subheader("Drag Polar Plot and Results")
        CL_vals = np.linspace(0, 1.5, 100); CD_vals = CDo + (CL_vals**2) / (np.pi * AR * e)
        CL_maxLD = np.sqrt(CDo * np.pi * AR * e); CD_maxLD = 2 * CDo
        max_LD = CL_maxLD / CD_maxLD if CL_maxLD > 0 else 0
        CD_point = CDo + (CL_point**2) / (np.pi * AR * e)
        LD_point = CL_point / CD_point if CL_point > 0 else 0
        
        c1, c2 = st.columns(2)
        c1.metric(label=f"CD at CL={CL_point:.2f}", value=f"{CD_point:.4f}")
        c2.metric(label=f"L/D at CL={CL_point:.2f}", value=f"{LD_point:.2f}")
        
        fig, ax = plt.subplots()
        ax.plot(CD_vals, CL_vals, 'b-', label='Drag Polar')
        ax.plot(CD_point, CL_point, 'ro', markersize=8, label=f'Operating Point (CL={CL_point:.2f})')
        ax.plot(CD_maxLD, CL_maxLD, 'gd', markersize=8, label=f'Max L/D = {max_LD:.2f}')
        ax.set_xlabel("Drag Coefficient (CD)"); ax.set_ylabel("Lift Coefficient (CL)")
        ax.set_title("Aeroplane Drag Polar"); ax.grid(True); ax.legend()
        ax.set_xlim(left=0); ax.set_ylim(bottom=0); st.pyplot(fig)

        st.subheader("Absolute Values at Operating Point")
        st.latex("L = C_L \\frac{1}{2} \\rho V^2 S \quad | \quad D = C_D \\frac{1}{2} \\rho V^2 S")
        q_w1 = 0.5 * rho_w1 * V_w1**2
        L_point_abs = CL_point * q_w1 * S_w1
        D_point_abs = CD_point * q_w1 * S_w1
        c3, c4 = st.columns(2)
        c3.metric(label="Lift (L)", value=f"{L_point_abs:.1f} N")
        c4.metric(label="Drag (D)", value=f"{D_point_abs:.1f} N")

# --- WEEK 2: ISENTROPIC FLOW TAB ---
with tab_week2:
    st.header("Isentropic Flow Calculator (Sessions 2a, 2b, 2c)")
    st.latex("\\frac{p_0}{p} = \\left(1 + \\frac{k-1}{2} Ma^2\\right)^{\\frac{k}{k-1}} \quad | \quad \\frac{T_0}{T} = 1 + \\frac{k-1}{2} Ma^2 \quad | \quad \\frac{A}{A^*} = \\frac{1}{Ma} \\left[ \\frac{2}{k+1} \\left(1 + \\frac{k-1}{2} Ma^2 \\right) \\right]^{\\frac{k+1}{2(k-1)}}")
    col1, col2 = st.columns([1, 2])
    with col1:
        k_isen = st.number_input("Specific Heat Ratio (γ or k)", value=1.4, format="%.3f", key="k_isen_w2")
        known_type = st.selectbox("Known Value",('Mach Number', 'p0/p', 'T0/T', 'rho0/rho', 'A/A* (subsonic)', 'A/A* (supersonic)'), key="known_type_w2")
        known_val = st.number_input(f"Enter Value for {known_type}", value=1.0, format="%.4f", key="known_val_w2")
        if st.button("Calculate Isentropic Relations", key="calc_isen_w2"):
            Ma = None
            if known_type == 'Mach Number': Ma = known_val
            elif 'A/A*' in known_type:
                if known_val < 1.0: st.error("A/A* must be >= 1.")
                else:
                    flow_regime = 'subsonic' if 'subsonic' in known_type else 'supersonic'
                    Ma = solve_for_ma(known_val, "A/A*", k_isen, flow_regime)
            else:
                Ma = solve_for_ma(known_val, known_type, k_isen, 'subsonic')
                if Ma is None or np.isnan(Ma): Ma = solve_for_ma(known_val, known_type, k_isen, 'supersonic')
            if Ma is not None and not np.isnan(Ma): st.session_state.isentropic_results = calculate_isentropic_ratios(Ma, k_isen)
            else:
                st.error("Could not solve for a valid Mach number."); st.session_state.isentropic_results = None
    with col2:
        st.subheader("Results - Ratios")
        if 'isentropic_results' in st.session_state and st.session_state.isentropic_results:
            res = st.session_state.isentropic_results
            st.dataframe({ "Parameter": res.keys(), "Value": [f"{v:.4f}" for v in res.values()] }, hide_index=True)
            
            st.markdown("---")
            st.subheader("Calculate Absolute Values from Ratios")
            p0_isen_abs = st.number_input("Stagnation Pressure (p0), Pa", value=101325.0, format="%.1f", key="p0_isen_abs")
            T0_isen_abs = st.number_input("Stagnation Temperature (T0), K", value=288.15, format="%.2f", key="T0_isen_abs")
            R_isen_abs = st.number_input("Gas Constant (R), J/kg·K", value=287.0, format="%.1f", key="R_isen_abs")
            
            p_abs = p0_isen_abs / res['p0/p']
            T_abs = T0_isen_abs / res['T0/T']
            rho_abs = p_abs / (R_isen_abs * T_abs)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Static Pressure (p)", f"{p_abs:.1f} Pa | {p_abs * PA_TO_ATM:.4f} atm")
            c2.metric("Static Temperature (T)", f"{T_abs:.2f} K | {T_abs + K_TO_C:.2f} °C")
            c3.metric("Static Density (ρ)", f"{rho_abs:.4f} kg/m³")
        else: st.info("Enter a value and click calculate.")

# --- WEEK 3: FANNO FLOW & WIND TUNNELS TAB ---
with tab_week3:
    st.header("Fanno Flow (Adiabatic Flow with Friction) (Session 3a)")
    st.latex("\\frac{f L^*}{D} = \\frac{1-Ma^2}{k Ma^2} + \\frac{k+1}{2k} \\ln \\left( \\frac{(k+1)Ma^2}{2 + (k-1)Ma^2} \\right)")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Inputs")
        k_fanno = st.number_input("Specific Heat Ratio (γ or k)", value=1.4, format="%.3f", key="k_fanno_w3")
        Ma1_fanno = st.number_input("Inlet Mach Number (Ma1)", value=0.4, format="%.4f", key="ma1_fanno_w3")
        f_fanno = st.number_input("Friction Factor (f)", value=0.0148, format="%.4f", key="f_fanno_w3")
        L_fanno = st.number_input("Duct Length (L) [m]", value=4.0, format="%.3f", key="L_fanno_w3")
        D_fanno = st.number_input("Duct Diameter (D) [m]", value=0.03, format="%.3f", key="D_fanno_w3")
        p1_fanno = st.number_input("Inlet Pressure (p1), Pa", value=150000.0, format="%.1f", key="p1_fanno_w3")
        T1_fanno = st.number_input("Inlet Temperature (T1), K", value=300.0, format="%.2f", key="T1_fanno_w3")

        if st.button("Calculate Fanno Flow", key="calc_fanno_w3"):
            if Ma1_fanno <= 0:
                st.error("Inlet Mach number must be > 0."); st.session_state.fanno_results = None
            else:
                fL_D_actual = f_fanno * L_fanno / D_fanno
                inlet_ratios = calculate_fanno_ratios(Ma1_fanno, k_fanno)
                
                if inlet_ratios is None:
                    st.error("Could not calculate inlet Fanno ratios.")
                    st.session_state.fanno_results = None
                else:
                    fL_star_D1 = inlet_ratios['fL*/D']
                    results = {'Ma1': Ma1_fanno, 'L_star': fL_star_D1 * D_fanno / f_fanno}

                    if fL_D_actual >= fL_star_D1:
                        results['choked'] = True; results['Ma2'] = 1.0
                    else:
                        results['choked'] = False
                        fL_star_D2 = fL_star_D1 - fL_D_actual
                        flow_regime = 'subsonic' if Ma1_fanno < 1 else 'supersonic'
                        results['Ma2'] = solve_ma_from_fanno_length(fL_star_D2, k_fanno, flow_regime)

                    if results.get('Ma2') is not None and not np.isnan(results['Ma2']):
                        exit_ratios = calculate_fanno_ratios(results['Ma2'], k_fanno)
                        if exit_ratios:
                            results['p2/p1'] = exit_ratios['p/p*'] / inlet_ratios['p/p*']
                            results['T2/T1'] = exit_ratios['T/T*'] / inlet_ratios['T/T*']
                            results['p02/p01'] = exit_ratios['p0/p0*'] / inlet_ratios['p0/p0*']
                    
                    st.session_state.fanno_results = results
    
    with col2:
        st.subheader("Results")
        if 'fanno_results' in st.session_state and st.session_state.fanno_results:
            res = st.session_state.fanno_results
            st.metric("Max Duct Length for Choking (L*)", f"{res['L_star']:.3f} m")
            if res.get('choked'):
                st.warning(f"Flow is CHOKED. Provided L ({L_fanno}m) >= L*.")
            else:
                st.success("Flow is NOT choked at the exit.")
            st.metric("Exit Mach Number (Ma2)", f"{res.get('Ma2', np.nan):.4f}")
            st.markdown("---")
            st.write("Ratios of Exit to Inlet Conditions:")
            data = {"Ratio": ["p2/p1", "T2/T1", "p0,2/p0,1"],
                    "Value": [f"{res.get('p2/p1', np.nan):.4f}", f"{res.get('T2/T1', np.nan):.4f}", f"{res.get('p02/p01', np.nan):.4f}"]}
            st.dataframe(data, hide_index=True)

            st.subheader("Absolute Exit Conditions")
            p2_fanno = p1_fanno * res.get('p2/p1', np.nan)
            T2_fanno = T1_fanno * res.get('T2/T1', np.nan)
            c1_fanno, c2_fanno = st.columns(2)
            c1_fanno.metric("Exit Pressure (p2)", f"{p2_fanno:.1f} Pa | {p2_fanno * PA_TO_ATM:.4f} atm")
            c2_fanno.metric("Exit Temperature (T2)", f"{T2_fanno:.2f} K | {T2_fanno + K_TO_C:.2f} °C")
        else:
            st.info("Enter duct parameters and click calculate.")
            
    st.markdown("---")
    st.header("Supersonic Wind Tunnel Diffuser Sizing (Session 3b)")
    st.latex("\\frac{A_{t2}}{A_{t1}} = \\frac{p_{01}}{p_{02}}")
    k_wt = st.number_input("Specific Heat Ratio (γ or k)", value=1.4, format="%.3f", key="k_wt_w3")
    ma_test = st.number_input("Test Section Mach Number", value=2.5, format="%.4f", key="ma_test_w3")
    At1_wt = st.number_input("Nozzle Throat Area (A_t1), m² (optional)", value=0.1, format="%.4f", key="At1_wt")
    if st.button("Calculate Diffuser Area Ratio", key="calc_wt_w3"):
        if ma_test <= 1.0: st.error("Test section Mach number must be > 1.")
        else:
            shock_res = calculate_normal_shock_relations(ma_test, k_wt)
            p02_p01 = shock_res['p02/p01']
            area_ratio = 1 / p02_p01
            st.metric("Required Diffuser Throat Ratio (A_t2 / A_t1)", f"{area_ratio:.4f}")
            if At1_wt > 0:
                At2_wt = area_ratio * At1_wt
                st.metric("Calculated Diffuser Throat Area (A_t2)", f"{At2_wt:.4f} m²")
            st.info("This assumes a normal shock at the diffuser entrance for tunnel starting.")

# --- WEEK 4: SHOCKS & MEASUREMENTS TAB ---
with tab_week4:
    st.header("Normal Shock Relations (Session 4a)")
    st.latex("M_2^2 = \\frac{M_1^2 + \\frac{2}{k-1}}{\\frac{2k}{k-1}M_1^2 - 1} \quad | \quad \\frac{p_2}{p_1} = 1 + \\frac{2k}{k+1}(M_1^2 - 1)")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        k_shock = st.number_input("Specific Heat Ratio (γ or k)", value=1.4, format="%.3f", key="k_shock_w4")
        m1_shock = st.number_input("Upstream Mach Number (M1)", value=2.0, format="%.4f", key="m1_shock_w4")
        p1_shock = st.number_input("Upstream Static Pressure (p1), Pa", value=101325.0, format="%.1f", key="p1_shock_w4")
        T1_shock = st.number_input("Upstream Static Temperature (T1), K", value=288.15, format="%.2f", key="T1_shock_w4")

        if st.button("Calculate Shock Relations", key="calc_shock_w4"):
            if m1_shock <= 1.0:
                st.error("Upstream Mach number (M1) must be > 1.")
                st.session_state.shock_results_w4 = None
                st.session_state.shock_absolute_results_w4 = None
            else:
                st.session_state.shock_results_w4 = calculate_normal_shock_relations(m1_shock, k_shock)
                # Also calculate absolute values
                p01 = p1_shock * (1 + (k_shock - 1) / 2 * m1_shock**2)**(k_shock / (k_shock - 1))
                p02_p01_ratio = st.session_state.shock_results_w4['p02/p01']
                p02 = p01 * p02_p01_ratio
                pressure_loss = p01 - p02
                
                T0 = T1_shock * (1 + (k_shock - 1) / 2 * m1_shock**2) # T0 is constant
                T2 = T1_shock * st.session_state.shock_results_w4['T2/T1']

                st.session_state.shock_absolute_results_w4 = {
                    'p01': p01,
                    'p02': p02,
                    'loss': pressure_loss,
                    'T0': T0,
                    'T2': T2,
                }

    with col2:
        st.subheader("Results - Ratios")
        if 'shock_results_w4' in st.session_state and st.session_state.shock_results_w4:
            res = st.session_state.shock_results_w4
            st.dataframe(
                { "Parameter": res.keys(), "Value": [f"{v:.4f}" for v in res.values()] },
                hide_index=True
            )
            
            st.subheader("Results - Absolute Values")
            abs_res = st.session_state.shock_absolute_results_w4
            c1_abs, c2_abs = st.columns(2)
            c1_abs.metric("Upstream Total Pressure (p01)", f"{abs_res['p01']:.1f} Pa | {abs_res['p01'] * PA_TO_ATM:.4f} atm")
            c2_abs.metric("Downstream Total Pressure (p02)", f"{abs_res['p02']:.1f} Pa | {abs_res['p02'] * PA_TO_ATM:.4f} atm")
            st.metric("Total Pressure Loss (p01 - p02)", f"{abs_res['loss']:.1f} Pa | {abs_res['loss'] * PA_TO_ATM:.4f} atm", delta=f"{-abs_res['loss']:.1f} Pa", delta_color="inverse")
            
            c3_abs, c4_abs = st.columns(2)
            c3_abs.metric("Stagnation Temperature (T0)", f"{abs_res['T0']:.2f} K | {abs_res['T0'] + K_TO_C:.2f} °C")
            c4_abs.metric("Downstream Static Temperature (T2)", f"{abs_res['T2']:.2f} K | {abs_res['T2'] + K_TO_C:.2f} °C")

        else:
            st.info("Enter upstream conditions and click calculate.")
            
    st.markdown("---")
    st.header("Pitot Tube Mach Number Finder (Session 4b)")
    st.latex("\\text{Subsonic:} \\frac{p_{01}}{p_1} = \\left(1 + \\frac{k-1}{2} Ma_1^2\\right)^{\\frac{k}{k-1}}")
    st.latex("\\text{Supersonic (Rayleigh Pitot):} \\frac{p_{02}}{p_1} = \\left[ \\frac{(\\frac{k+1}{2}Ma_1^2)^k}{(\\frac{2k}{k+1}Ma_1^2 - \\frac{k-1}{k+1})} \\right]^{\\frac{1}{k-1}}")
    k_pitot = st.number_input("Specific Heat Ratio (γ or k)", value=1.4, format="%.3f", key="k_pitot_w4")
    p1_pitot = st.number_input("Static Pressure (p1)", value=101325.0, format="%.2f", key="p1_pitot_w4")
    p_pitot = st.number_input("Pitot Pressure (measured)", value=129000.0, format="%.2f", key="p_pitot_w4")
    if st.button("Calculate Mach Number", key="calc_pitot_w4"):
        if p_pitot <= p1_pitot: st.error("Pitot pressure must be greater than static pressure.")
        else:
            ratio = p_pitot / p1_pitot
            critical_p_ratio = (1 + (k_pitot - 1) / 2)**(k_pitot / (k_pitot - 1))
            if ratio < critical_p_ratio:
                st.subheader("Flow Regime: Subsonic"); Ma = solve_for_ma(ratio, 'p0/p', k_pitot, 'subsonic')
            else:
                st.subheader("Flow Regime: Supersonic"); Ma = solve_rayleigh_pitot(ratio, k_pitot)
            if Ma is not None and not np.isnan(Ma): st.metric(label="Calculated Mach Number (Ma)", value=f"{Ma:.4f}")
            else: st.error("Failed to calculate Mach number.")

# --- WEEK 5: OBLIQUE SHOCKS TAB ---
with tab_week5:
    st.header("Oblique Shocks & Expansion Waves (Session 5)")
    st.markdown("Calculators based on Session 5a (Intro), 5b (Relations), and 5c (Examples).")

    # --- Mach Angle Calculator ---
    with st.expander("**Mach Angle (Session 5a & 5c)**: `μ = asin(1 / Ma)`"):
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"\mu = \arcsin\left(\frac{1}{Ma}\right)")
            solve_for_mach_angle = st.radio("Solve for:", ('Mach Angle (μ)', 'Mach Number (Ma)'), key="mach_angle_solve")
            Ma_in_angle = st.number_input("Mach Number (Ma)", value=2.0, min_value=1.0001, format="%.4f", key="Ma_in_angle", disabled=(solve_for_mach_angle=='Mach Number (Ma)'))
            mu_in_angle = st.number_input("Mach Angle (μ) in degrees", value=30.0, min_value=0.0, max_value=90.0, format="%.2f", key="mu_in_angle", disabled=(solve_for_mach_angle=='Mach Angle (μ)'))
        with col2:
            st.subheader("Result")
            try:
                if solve_for_mach_angle == 'Mach Angle (μ)':
                    if Ma_in_angle <= 1.0: st.error("Mach number must be > 1 for a Mach wave.")
                    else: 
                        mu_rad = np.arcsin(1 / Ma_in_angle)
                        mu_deg = np.degrees(mu_rad)
                        st.metric("Mach Angle (μ)", f"{mu_deg:.2f}°")
                else: # Solve for Ma
                    if mu_in_angle <= 0 or mu_in_angle >= 90: st.error("Angle must be > 0° and < 90°.")
                    else: 
                        mu_rad = np.radians(mu_in_angle)
                        Ma_out = 1 / np.sin(mu_rad)
                        st.metric("Mach Number (Ma)", f"{Ma_out:.4f}")
            except Exception as e: st.error(f"Calculation failed: {e}")

    # --- Theta-Beta-Mach Relation ---
    with st.expander("**Theta-Beta-Mach Relation (Session 5b)**"):
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"\tan \theta = 2 \cot \beta \frac{Ma_1^2 \sin^2 \beta - 1}{Ma_1^2 (k + \cos 2\beta) + 2}")
            st.markdown("Calculates deflection angle (θ) from $Ma_1$ and shock angle (β).")
            k_tbm = st.number_input("k", value=1.4, format="%.3f", key="k_tbm")
            Ma1_tbm = st.number_input("Upstream Mach (Ma1)", value=2.4, min_value=1.0001, format="%.4f", key="Ma1_tbm")
            beta_tbm = st.number_input("Shock Angle (β) in degrees", value=30.0, min_value=0.0, max_value=90.0, format="%.2f", key="beta_tbm")
        with col2:
            st.subheader("Result")
            try:
                mu_rad = np.arcsin(1 / Ma1_tbm); mu_deg = np.degrees(mu_rad)
                if beta_tbm <= mu_deg: 
                    st.error(f"Shock angle (β) must be > Mach angle (μ = {mu_deg:.2f}°).")
                elif Ma1_tbm <= 1.0:
                    st.error("Upstream Mach Number (Ma1) must be > 1.")
                else:
                    theta_deg = get_theta_from_beta(beta_tbm, Ma1_tbm, k_tbm)
                    st.metric("Deflection Angle (θ)", f"{theta_deg:.2f}°")
                    st.info(f"For $Ma_1={Ma1_tbm}$, the minimum shock angle (Mach Angle μ) is {mu_deg:.2f}°. A normal shock is β=90°.")
            except Exception as e: st.error(f"Calculation failed: {e}")

    # --- Full Oblique Shock Solver ---
    with st.expander("**Oblique Shock Relations Solver (Session 5b & 5c)**"):
        st.markdown("This tool calculates all downstream properties from the upstream Mach number ($Ma_1$) and the shock wave angle (β), following the method in Session 5c, Examples 2 & 3.")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Inputs")
            k_ob = st.number_input("k", value=1.4, format="%.3f", key="k_ob")
            Ma1_ob = st.number_input("Upstream Mach (Ma1)", value=2.0, min_value=1.0001, format="%.4f", key="Ma1_ob")
            beta_ob = st.number_input("Shock Angle (β) in degrees", value=53.4, format="%.2f", key="beta_ob")
            
            st.markdown("---")
            st.subheader("Upstream Absolute Values (Optional)")
            p1_ob = st.number_input("Upstream Static Pressure (p1), Pa", value=101325.0, format="%.1f", key="p1_ob")
            T1_ob = st.number_input("Upstream Static Temperature (T1), K", value=288.0, format="%.2f", key="T1_ob")

            if st.button("Calculate Oblique Shock Relations", key="calc_ob_w5"):
                try:
                    results = get_oblique_shock_properties(Ma1_ob, beta_ob, k_ob)
                    if results is None:
                        mu_deg = np.degrees(np.arcsin(1/Ma1_ob))
                        st.error(f"Calculation failed. Check inputs. Is Ma1 > 1? Is β > Mach Angle (μ = {mu_deg:.2f}°)?")
                        st.session_state.oblique_results_w5 = None
                    else:
                        st.session_state.oblique_results_w5 = results
                        
                        # Calculate absolute values
                        p2_abs = p1_ob * results['p2/p1']
                        T2_abs = T1_ob * results['T2/T1']
                        
                        isen_res_1 = calculate_isentropic_ratios(Ma1_ob, k_ob)
                        p01_abs = p1_ob * isen_res_1['p0/p']
                        T01_abs = T1_ob * isen_res_1['T0/T']
                        
                        p02_abs = p01_abs * results['p02/p01']
                        T02_abs = T01_abs # T0 is constant
                        
                        st.session_state.oblique_abs_results_w5 = {
                            'p1': p1_ob, 'T1': T1_ob, 'p2': p2_abs, 'T2': T2_abs,
                            'p01': p01_abs, 'T01': T01_abs, 'p02': p02_abs, 'T02': T02_abs,
                            'loss': p01_abs - p02_abs
                        }
                except Exception as e:
                    st.error(f"Calculation Failed. Check inputs. Error: {e}")
                    st.session_state.oblique_results_w5 = None

        with col2:
            st.subheader("Results - Ratios")
            if 'oblique_results_w5' in st.session_state and st.session_state.oblique_results_w5:
                res = st.session_state.oblique_results_w5
                st.dataframe(
                    { "Parameter": res.keys(), "Value": [f"{v:.4f}" for v in res.values()] },
                    hide_index=True, use_container_width=True
                )
                
                st.subheader("Results - Absolute Values")
                abs_res = st.session_state.oblique_abs_results_w5
                c1_abs, c2_abs = st.columns(2)
                c1_abs.metric("Downstream Static Pressure (p2)", f"{abs_res['p2']:.1f} Pa | {abs_res['p2'] * PA_TO_ATM:.4f} atm")
                c2_abs.metric("Downstream Static Temp (T2)", f"{abs_res['T2']:.2f} K | {abs_res['T2'] + K_TO_C:.2f} °C")
                
                c3_abs, c4_abs = st.columns(2)
                c3_abs.metric("Upstream Total Pressure (p01)", f"{abs_res['p01']:.1f} Pa | {abs_res['p01'] * PA_TO_ATM:.4f} atm")
                c4_abs.metric("Downstream Total Pressure (p02)", f"{abs_res['p02']:.1f} Pa | {abs_res['p02'] * PA_TO_ATM:.4f} atm")
                
                st.metric("Total Pressure Loss (p01 - p02)", f"{abs_res['loss']:.1f} Pa", delta=f"{-abs_res['loss']:.1f} Pa", delta_color="inverse")
                st.metric("Total Temperature (T01 = T02)", f"{abs_res['T01']:.2f} K | {abs_res['T01'] + K_TO_C:.2f} °C")

            else:
                st.info("Enter upstream conditions and click calculate.")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SECTION 3: NEW WEEK 6 TAB
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

with tab_week6:
    st.header("θ-β-Ma Diagram & Shock Reflections (Session 6)")
    st.markdown("Calculators based on Session 6a (TBM Diagram), 6b (Reflections), and 6d (Examples).")

    # --- Detached Shock Advisor ---
    with st.expander("**Detached Shock Advisor (Session 6a)**"):
        st.markdown("Calculates the maximum deflection angle ($\theta_{max}$) for a given $Ma_1$. If your wedge angle is greater than this, a detached shock will form.")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"\theta > \theta_{max} \implies \text{Detached Shock}")
            k_det = st.number_input("k", value=1.4, format="%.3f", key="k_det")
            Ma1_det = st.number_input("Upstream Mach (Ma1)", value=2.0, min_value=1.0001, format="%.4f", key="Ma1_det")
            theta_det = st.number_input("Your Deflection Angle (θ) in degrees", value=20.0, format="%.2f", key="theta_det")
        with col2:
            st.subheader("Result")
            if Ma1_det <= 1.0:
                st.error("Ma1 must be > 1.0")
            else:
                theta_max, beta_at_max = get_max_theta(Ma1_det, k_det)
                st.metric("Max Deflection Angle (θ_max)", f"{theta_max:.2f}°")
                st.metric("Shock Angle at θ_max (β)", f"{beta_at_max:.2f}°")
                
                if theta_det > theta_max:
                    st.error(f"Detached Shock: Your angle ({theta_det}°) > θ_max ({theta_max:.2f}°)")
                else:
                    st.success(f"Attached Shock Possible: Your angle ({theta_det}°) ≤ θ_max ({theta_max:.2f}°)")

    # --- Weak vs. Strong Solution Finder ---
    with st.expander("**Weak vs. Strong Solution Finder (Session 6a & 6d)**"):
        st.markdown("Finds the two possible shock angles (weak and strong) for a given $Ma_1$ and $\theta$, as shown in the $\theta-\beta-Ma$ diagram.")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"\text{Find } \beta \text{ from } Ma_1 \text{ and } \theta")
            k_sol = st.number_input("k", value=1.4, format="%.3f", key="k_sol")
            Ma1_sol = st.number_input("Upstream Mach (Ma1)", value=2.0, min_value=1.0001, format="%.4f", key="Ma1_sol")
            theta_sol = st.number_input("Deflection Angle (θ) in degrees", value=10.0, format="%.2f", key="theta_sol")
        with col2:
            st.subheader("Results")
            if Ma1_sol <= 1.0:
                st.error("Ma1 must be > 1.0")
            else:
                beta_weak, beta_strong, theta_max = solve_beta(Ma1_sol, theta_sol, k_sol)
                st.metric("Max Deflection (θ_max)", f"{theta_max:.2f}°")
                if np.isnan(beta_weak):
                    st.error(f"Detached Shock: Your angle ({theta_sol}°) > θ_max ({theta_max:.2f}°)")
                else:
                    st.metric("Weak Solution (β_weak)", f"{beta_weak:.2f}°")
                    st.metric("Strong Solution (β_strong)", f"{beta_strong:.2f}°")

    # --- Shock Reflection Solver ---
    with st.expander("**Shock Reflection Solver (Session 6b & 6d)**"):
        st.markdown("Solves the full shock reflection problem from a concave corner and a flat wall, as shown in Session 6d, Example 2.")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Inputs")
            k_ref = st.number_input("k", value=1.4, format="%.3f", key="k_ref")
            st.markdown("---")
            st.subheader("Region 1 (Freestream)")
            Ma1_ref = st.number_input("Ma1", value=3.6, min_value=1.0001, format="%.4f", key="Ma1_ref")
            theta_ref = st.number_input("Deflection Angle (θ) in degrees", value=10.0, format="%.2f", key="theta_ref")
            p1_ref = st.number_input("p1 (Pa)", value=101000.0, format="%.1f", key="p1_ref")
            T1_ref = st.number_input("T1 (K)", value=288.0, format="%.2f", key="T1_ref")
            
            if st.button("Calculate Shock Reflection", key="calc_ref_w6"):
                st.session_state.ref_results = {}
                # --- Step 1: Incident Shock ---
                beta_1, _, theta_max_1 = solve_beta(Ma1_ref, theta_ref, k_ref)
                if np.isnan(beta_1):
                    st.error(f"Incident shock is detached: θ ({theta_ref}°) > θ_max ({theta_max_1:.2f}°)")
                    st.session_state.ref_results = None
                else:
                    props_1 = get_oblique_shock_properties(Ma1_ref, beta_1, k_ref)
                    Ma2 = props_1['Ma2']
                    p2 = p1_ref * props_1['p2/p1']
                    T2 = T1_ref * props_1['T2/T1']
                    st.session_state.ref_results['region2'] = {'Ma2': Ma2, 'p2': p2, 'T2': T2, 'beta1': beta_1}
                    
                    # --- Step 2: Reflected Shock ---
                    # Upstream conditions are now Ma2 and theta_ref
                    beta_2, _, theta_max_2 = solve_beta(Ma2, theta_ref, k_ref)
                    if np.isnan(beta_2):
                        st.error(f"Reflected shock is detached: θ ({theta_ref}°) > θ_max for Ma2 ({theta_max_2:.2f}°)")
                        st.session_state.ref_results['region3'] = None
                    else:
                        props_2 = get_oblique_shock_properties(Ma2, beta_2, k_ref)
                        Ma3 = props_2['Ma2']
                        p3 = p2 * props_2['p2/p1']
                        T3 = T2 * props_2['T2/T1']
                        phi = beta_2 - theta_ref # Reflected angle to wall
                        st.session_state.ref_results['region3'] = {'Ma3': Ma3, 'p3': p3, 'T3': T3, 'beta2': beta_2, 'phi': phi}
        
        with col2:
            st.subheader("Results")
            if 'ref_results' in st.session_state and st.session_state.ref_results:
                res2 = st.session_state.ref_results.get('region2')
                if res2:
                    st.subheader("Region 2 (After Incident Shock)")
                    c1, c2 = st.columns(2)
                    c1.metric("Incident Shock Angle (β1)", f"{res2['beta1']:.2f}°")
                    c2.metric("Ma2", f"{res2['Ma2']:.4f}")
                    c1.metric("p2", f"{res2['p2']:.1f} Pa | {res2['p2'] * PA_TO_ATM:.4f} atm")
                    c2.metric("T2", f"{res2['T2']:.2f} K | {res2['T2'] + K_TO_C:.2f} °C")
                
                res3 = st.session_state.ref_results.get('region3')
                if res3:
                    st.markdown("---")
                    st.subheader("Region 3 (After Reflected Shock)")
                    c3, c4 = st.columns(2)
                    c3.metric("Reflected Shock Angle (β2)", f"{res3['beta2']:.2f}°")
                    c4.metric("Ma3", f"{res3['Ma3']:.4f}")
                    c3.metric("p3", f"{res3['p3']:.1f} Pa | {res3['p3'] * PA_TO_ATM:.4f} atm")
                    c4.metric("T3", f"{res3['T3']:.2f} K | {res3['T3'] + K_TO_C:.2f} °C")
                    st.metric("Reflected Angle to Wall (Φ)", f"{res3['phi']:.2f}°")
            else:
                st.info("Enter freestream conditions and click calculate.")
