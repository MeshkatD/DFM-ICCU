
import numpy as np
import math as ma
from scipy.optimize import fsolve


def simulate_hydrogenation_model(
    k7, k8, k9, k10,
    E7, E8, E10, T, P_hyd, n,
    D, L, Time_hyd, u, epsilon,
    C_feed_hyd_H2, C_feed_hyd_N2, C_CO2_init_hyd, C_H2O_init_hyd,
    theta_CO2_init_hyd, theta_H2O_init_hyd,
    rho, Omega, Nx, Nt
):
    dx = L / (Nx - 1)
    dt = Time_hyd / Nt
    R_ = 0.08206  # (L.atm / mol.K)

    # Allocate solution arrays
    C_CO2 = np.zeros((Nt + 1, Nx))
    C_H2 = np.zeros((Nt + 1, Nx))
    C_H2O = np.zeros((Nt + 1, Nx))
    C_CH4 = np.zeros((Nt + 1, Nx))
    C_N2 = np.zeros((Nt + 1, Nx))
    theta_CO2 = np.zeros((Nt + 1, Nx))
    theta_H2O = np.zeros((Nt + 1, Nx))

    # Constants for Ergun Equation (to be checked later)
    R_gas   = 8.314             # J/(mol·K)
    M_mix   = 0.028             # Approx. Molar Mass for 10% H2 / 90% N2
    mu      = 3.0e-5            # Pa·s, H2/N2 mix viscosity approx.
    dp      = 1e-3              # m, particle diameter (1 mm)

    # Convert inlet pressure from atm to Pa
    P0 = P_hyd * 101325.0       # Pa
    C_init = P0/(R_gas * T) * 1e-3

    # Initial conditions at t=0
    C_CO2[0, :] = C_CO2_init_hyd
    C_H2O[0, :] = C_H2O_init_hyd
    C_CH4[0, :] = 0
    C_N2[0, :] = C_init - (C_CO2_init_hyd + C_H2O_init_hyd)
    C_H2[0, 0] = (C_feed_hyd_H2 / 100) * P_hyd / (R_ * T)
    theta_CO2[0, :] = theta_CO2_init_hyd
    theta_H2O[0, :] = theta_H2O_init_hyd

    # Precompute Ergun pressure drop, local velocity, and fluid density
    P_grid = np.zeros(Nx)
    u_grid = np.zeros(Nx)
    rho_grid = np.zeros(Nx)
    P_grid[0] = P0
    u_grid[0] = u
    rho_grid[0] = (P_grid[0] / (R_gas * T)) * M_mix

    for j in range(Nx - 1):
        rho_mol = P_grid[j] / (R_gas * T)
        rho_fl = rho_mol * M_mix
        rho_grid[j] = rho_fl 

        u_m_s = u_grid[j] / 100.0           # m/s
        term1 = 150 * mu * (1 - epsilon)**2 / (epsilon**3 * dp**2) * u_m_s
        term2 = 1.75 * rho_fl * (u_m_s**2) * (1 - epsilon) / (epsilon**3 * dp)
        dPdz = -(term1 + term2)
        P_grid[j + 1] = P_grid[j] + dPdz * (dx / 100.0)
        u_grid[j + 1] = u * (P0 / P_grid[j + 1])
        
    rho_grid[-1] = (P_grid[-1] / (R_gas * T)) * M_mix 

    # Equilibrium constant
    K_eq = np.exp(0.5032 * ((56000 / T**2) + (34633 / T) - (16.4 * np.log(T)) + (0.00557 * T)) + 33.165)

    # Pre-calculate rate constants
    k7_exp = k7 * np.exp(-E7 / (R_gas/1000 * T))
    k8_exp = k8 * np.exp(-E8 / (R_gas/1000 * T))
    k10_exp = k10 * np.exp(-E10 / (R_gas/1000 * T))

    # Residuals for implicit step
    def backward_euler_equations(y, i, t):
        C_CO2_next, C_H2_next, C_CH4_next, C_H2O_next, C_N2_next, theta_CO2_next, theta_H2O_next = y

        P_local_atm = P_grid[i] / 101325.0
        C_total_next = sum(y[:5])
        if C_total_next < 1e-12: C_total_next = 1e-12
        
        P_CO2_next = (C_CO2_next / C_total_next) * P_local_atm
        P_H2_next = (C_H2_next / C_total_next) * P_local_atm
        P_CH4_next = (C_CH4_next / C_total_next) * P_local_atm
        P_H2O_next = (C_H2O_next / C_total_next) * P_local_atm
        
        # Reaction rate calculations at time t+1
        # Rate of CO2 desorption
        r_CO2_des_next = k7_exp * theta_CO2_next * C_H2_next
        
        # Rate of formation of CH4
        Approach_to_Equilibrium_next = P_CO2_next * (P_H2_next**4) - (P_CH4_next * P_H2O_next**2) / K_eq
        
        absApproach_to_Equilibrium_next = abs(Approach_to_Equilibrium_next)

        if absApproach_to_Equilibrium_next <= 1e-6:
            r_CH4_hyd_next = k8_exp * ma.copysign((268851.797358742 - (124307820284.15 * absApproach_to_Equilibrium_next)) * absApproach_to_Equilibrium_next, Approach_to_Equilibrium_next)
        else:
            r_CH4_hyd_next = k8_exp * ma.copysign(absApproach_to_Equilibrium_next**n, Approach_to_Equilibrium_next)
        
        r_H2O_ads_next = k10_exp * C_H2O_next * (1 - theta_CO2_next - theta_H2O_next) - k9 * theta_H2O_next
        
        u_loc = u_grid[i]

        # Diffusion terms with Dankwert boundary conditions
        if i == 1:  # Outlet boundary (last node)
            diffusion_CO2 = (D * dt / epsilon) * (C_CO2[t, i + 1] - C_CO2_next) / dx**2
            diffusion_H2 = (D * dt / epsilon) * (C_H2[t, i + 1] - C_H2_next) / dx**2
            diffusion_CH4 = (D * dt / epsilon) * (C_CH4[t, i + 1] - C_CH4_next) / dx**2
            diffusion_H2O = (D * dt / epsilon) * (C_H2O[t, i + 1] - C_H2O_next) / dx**2
            diffusion_N2 = (D * dt / epsilon) * (C_N2[t, i + 1] - C_N2_next) / dx**2
        
        elif i == Nx - 1:  # Outlet boundary (last node)
            diffusion_CO2 = (D * dt / epsilon) * (-C_CO2_next + C_CO2[t + 1, i - 1]) / dx**2
            diffusion_H2 = (D * dt / epsilon) * (-C_H2_next + C_H2[t + 1, i - 1]) / dx**2
            diffusion_CH4 = (D * dt / epsilon) * (-C_CH4_next + C_CH4[t + 1, i - 1]) / dx**2
            diffusion_H2O = (D * dt / epsilon) * (-C_H2O_next + C_H2O[t + 1, i - 1]) / dx**2
            diffusion_N2 = (D * dt / epsilon) * (-C_N2_next + C_N2[t + 1, i - 1]) / dx**2
        else:  # Inlet and Internal nodes
            # Handles i=1 case correctly by using C_CO2[t, i+1]
            diffusion_CO2 = (D * dt / epsilon) * (C_CO2[t, i + 1] - 2 * C_CO2_next + C_CO2[t + 1, i - 1]) / dx**2
            diffusion_H2 = (D * dt / epsilon) * (C_H2[t, i + 1] - 2 * C_H2_next + C_H2[t + 1, i - 1]) / dx**2
            diffusion_CH4 = (D * dt / epsilon) * (C_CH4[t, i + 1] - 2 * C_CH4_next + C_CH4[t + 1, i - 1]) / dx**2
            diffusion_H2O = (D * dt / epsilon) * (C_H2O[t, i + 1] - 2 * C_H2O_next + C_H2O[t + 1, i - 1]) / dx**2
            diffusion_N2 = (D * dt / epsilon) * (C_N2[t, i + 1] - 2 * C_N2_next + C_N2[t + 1, i - 1]) / dx**2
            
        # Convection terms (upwind scheme)
        convection_CO2 = - (u_loc * dt / epsilon) * (C_CO2_next - C_CO2[t + 1, i - 1]) / dx
        convection_H2 = - (u_loc * dt / epsilon) * (C_H2_next - C_H2[t + 1, i - 1]) / dx
        convection_CH4 = - (u_loc * dt / epsilon) * (C_CH4_next - C_CH4[t + 1, i - 1]) / dx
        convection_H2O = - (u_loc * dt / epsilon) * (C_H2O_next - C_H2O[t + 1, i - 1]) / dx
        convection_N2 = - (u_loc * dt / epsilon) * (C_N2_next - C_N2[t + 1, i - 1]) / dx

        # --- Residual Equations ---
        # This now correctly uses 'rho' (bed density) passed into the main function
        eq1 = C_CO2_next - C_CO2[t, i] - diffusion_CO2 - convection_CO2 - (rho * (r_CO2_des_next - r_CH4_hyd_next) * dt) / epsilon
        eq2 = C_H2_next - C_H2[t, i] - diffusion_H2 - convection_H2 - (rho * (-4 * r_CH4_hyd_next) * dt) / epsilon
        eq3 = C_CH4_next - C_CH4[t, i] - diffusion_CH4 - convection_CH4 - (rho * r_CH4_hyd_next * dt) / epsilon
        eq4 = C_H2O_next - C_H2O[t, i] - diffusion_H2O - convection_H2O - (rho * (2 * r_CH4_hyd_next - r_H2O_ads_next) * dt) / epsilon
        eq5 = C_N2_next - C_N2[t, i] - diffusion_N2 - convection_N2 # Inert

        # Coverage factor updates
        eq6 = theta_CO2_next - theta_CO2[t, i] - (-r_CO2_des_next) * dt / Omega
        eq7 = theta_H2O_next - theta_H2O[t, i] - (r_H2O_ads_next) * dt / Omega

        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]

    # Time loop
    for t in range(0, Nt):

        # Boundary conditions at the inlet (i = 0)
        # These are set once per time step and are based on the constant inlet pressure P_hyd
        C_CO2[t+1, 0] = 0  # No CO2 is being fed
        C_H2[t+1, 0] = C_feed_hyd_H2 / 100 * P_hyd / (R_ * T)  # H2 is being fed at the inlet
        C_CH4[t+1, 0] = 0  # No CH4 is being fed
        C_H2O[t+1, 0] = 0  # No H2O is being fed
        C_N2[t+1, 0] = C_feed_hyd_N2 / 100 * P_hyd / (R_ * T)  # N2 is being fed at the inlet
        
        # Spatial loop (using backward Euler)
        for i in range(1, Nx):
            # Solve the implicit equations
            initial_guess = [C_CO2[t, i], C_H2[t, i], C_CH4[t, i], C_H2O[t, i], C_N2[t, i], theta_CO2[t, i], theta_H2O[t, i]]
            solution = fsolve(backward_euler_equations, initial_guess, args=(i,t))

            # Unpack the solution
            (C_CO2[t+1, i], C_H2[t+1, i], C_CH4[t+1, i], C_H2O[t+1, i], C_N2[t+1, i], 
             theta_CO2[t+1, i], theta_H2O[t+1, i]) = solution

    return C_CO2, C_CH4, C_H2, C_H2O, theta_CO2, theta_H2O, P_grid, u_grid, rho_grid

