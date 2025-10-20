
import numpy as np
from scipy.optimize import fsolve

# Function to simulate the model with purge of CO2 and H2O
def simulate_purge_model(
    k2, k3, k4, k5, k6, K_CO2, m,
    E6, T, P_prg, Alfa,
    D, L, Time_prg, u, epsilon,
    C_feed_purge_CO2, C_feed_purge_H2O, C_CO2_init_prg, C_H2O_init_prg,
    C_CH4_init_prg, C_H2_init_prg,
    theta_H2O_init_prg, theta_CO2_init_prg,
    rho, Omega, Nx, Nt
):
    dx = L / (Nx - 1)
    dt = Time_prg / Nt

    # Allocate solution arrays: time x space
    C_CO2 = np.zeros((Nt + 1, Nx))
    C_H2O = np.zeros((Nt + 1, Nx))
    C_CH4 = np.zeros((Nt + 1, Nx))
    C_H2 = np.zeros((Nt + 1, Nx))
    theta_CO2 = np.zeros((Nt + 1, Nx))
    theta_H2O = np.zeros((Nt + 1, Nx))

    # Constants for Ergun Equation
    R_gas   = 8.314             # J/(mol·K)
    M_mix   = 0.028             # kg/mol, approximate for N2 purge gas
    mu      = 3.0e-5            # Pa·s, N2 viscosity (350C)
    dp      = 1e-3              # m, particle diameter (1 mm)

    # Convert inlet pressure from atm to Pa
    P0 = P_prg * 101325.0       # Pa

    # Inlet concentration [mmol/mL] via ideal-gas at inlet pressure
    R_ = 0.08206  # (L.atm / mol.K)
    C0_CO2 = (C_feed_purge_CO2/100.0) * P_prg / (R_ * T)
    C0_H2O = (C_feed_purge_H2O/100.0) * P_prg / (R_ * T)

    # Initial profile (t=0) from previous stage
    C_CO2[0, :] = C_CO2_init_prg
    C_H2O[0, :] = C_H2O_init_prg
    C_CH4[0, :] = C_CH4_init_prg
    C_H2[0, :] = C_H2_init_prg
    theta_H2O[0, :] = theta_H2O_init_prg
    theta_CO2[0, :] = theta_CO2_init_prg

    # Precompute Ergun pressure drop and local velocity along bed
    P_grid = np.zeros(Nx)
    u_grid = np.zeros(Nx)
    rho_grid = np.zeros(Nx)
    P_grid[0] = P0
    u_grid[0] = u
    rho_grid[0] = (P_grid[0] / (R_gas * T)) * M_mix

    for j in range(Nx - 1):
        rho_mol = P_grid[j] / (R_gas * T)  # mol/m3
        rho_fl = rho_mol * M_mix          # kg/m3
        rho_grid[j] = rho_fl

        # Convert u from cm/s to m/s for Ergun calculation
        u_m_s = u_grid[j] / 100.0

        term1 = 150 * mu * (1 - epsilon)**2 / (epsilon**3 * dp**2) * u_m_s
        term2 = 1.75 * rho_fl * (u_m_s**2) * (1 - epsilon) / (epsilon**3 * dp)
        dPdz = -(term1 + term2)

        # Convert dx from cm to m for pressure update
        P_grid[j + 1] = P_grid[j] + dPdz * (dx / 100.0)
        
        # Keep u_grid in cm/s for convection term
        u_grid[j + 1] = u * (P0 / P_grid[j + 1])

    rho_grid[-1] = (P_grid[-1] / (R_gas * T)) * M_mix

    # Residuals for implicit step at (t,i)
    def backward_euler_equations(y, i, t):
        C_CO2_next, C_H2O_next, C_CH4_next, C_H2_next, theta_CO2_next, theta_H2O_next = y

        # Diffusion (central / one-sided at boundaries)
        if i == 1:
            diff_CO2 = (D*dt/epsilon)*(C_CO2[t,i+1] - C_CO2_next)/dx**2
            diff_H2O = (D*dt/epsilon)*(C_H2O[t,i+1] - C_H2O_next)/dx**2
            diff_CH4 = (D*dt/epsilon)*(C_CH4[t,i+1] - C_CH4_next)/dx**2
            diff_H2 = (D*dt/epsilon)*(C_H2[t,i+1] - C_H2_next)/dx**2
        elif i == Nx - 1:
            diff_CO2 = (D*dt/epsilon)*(-C_CO2_next + C_CO2[t+1,i-1])/dx**2
            diff_H2O = (D*dt/epsilon)*(-C_H2O_next + C_H2O[t+1,i-1])/dx**2
            diff_CH4 = (D*dt/epsilon)*(-C_CH4_next + C_CH4[t+1,i-1])/dx**2
            diff_H2 = (D*dt/epsilon)*(-C_H2_next + C_H2[t+1,i-1])/dx**2
        else:
            diff_CO2 = (D*dt/epsilon)*(C_CO2[t,i+1] - 2*C_CO2_next + C_CO2[t+1,i-1])/dx**2
            diff_H2O = (D*dt/epsilon)*(C_H2O[t,i+1] - 2*C_H2O_next + C_H2O[t+1,i-1])/dx**2
            diff_CH4 = (D*dt/epsilon)*(C_CH4[t,i+1] - 2*C_CH4_next + C_CH4[t+1,i-1])/dx**2
            diff_H2 = (D*dt/epsilon)*(C_H2[t,i+1] - 2*C_H2_next + C_H2[t+1,i-1])/dx**2

        # Convection
        u_loc = u_grid[i]
        conv_CO2 = -u_loc * dt / epsilon * (C_CO2_next - C_CO2[t+1, i - 1]) / dx
        conv_H2O = -u_loc * dt / epsilon * (C_H2O_next - C_H2O[t+1, i - 1]) / dx
        conv_CH4 = -u_loc * dt / epsilon * (C_CH4_next - C_CH4[t+1, i - 1]) / dx
        conv_H2 = -u_loc * dt / epsilon * (C_H2_next - C_H2[t+1, i - 1]) / dx

        # Scale concentrations by local pressure for reaction rates
        C2eff = C_CO2_next * (P_grid[i]/P0)
        Heff = C_H2O_next * (P_grid[i]/P0)

        # Reaction/Desorption Rates
        CO2_formation_rate = k6 * np.exp((-E6 / (R_gas/1000 * T)) * (1 - Alfa * theta_CO2_next)) * theta_CO2_next
        H2O_formation_rate = k2 * C2eff * theta_H2O_next - k3 * Heff * (1 - theta_CO2_next - theta_H2O_next) \
            - k4 * (Heff * theta_CO2_next / (1 + K_CO2 * C2eff**m))

        # Residual Equations
        eq1 = C_CO2_next - C_CO2[t, i] - diff_CO2 - conv_CO2 - (rho * CO2_formation_rate * dt / epsilon)
        eq2 = C_H2O_next - C_H2O[t, i] - diff_H2O - conv_H2O - (rho * H2O_formation_rate * dt / epsilon)
        eq3 = theta_CO2_next - theta_CO2[t, i] + (CO2_formation_rate * dt / Omega)
        eq4 = theta_H2O_next - theta_H2O[t, i] + (H2O_formation_rate * dt / Omega)
        eq5 = C_CH4_next - C_CH4[t, i] - diff_CH4 - conv_CH4
        eq6 = C_H2_next - C_H2[t, i] - diff_H2 - conv_H2

        return [eq1, eq2, eq3, eq4, eq5, eq6]

    # Time marching
    for t in range(0,Nt):
        # Inlet boundary at next time
        C_CO2[t + 1, 0] = C0_CO2
        C_H2O[t + 1, 0] = C0_H2O
        C_CH4[t + 1, 0] = 0
        C_H2[t + 1, 0] = 0

        # Solve interior nodes
        for i in range(1, Nx):
            guess = [C_CO2[t, i], C_H2O[t, i], C_CH4[t, i], C_H2[t, i], theta_CO2[t, i], theta_H2O[t, i]]
            sol = fsolve(backward_euler_equations, guess, args=(i, t))
            C_CO2[t + 1, i], C_H2O[t + 1, i], C_CH4[t + 1, i], C_H2[t + 1, i], theta_CO2[t + 1, i], theta_H2O[t + 1, i] = sol

        # Enforce physical bounds on coverages
        theta_CO2[t + 1, :] = np.clip(theta_CO2[t + 1, :], 0, 1)
        theta_H2O[t + 1, :] = np.clip(theta_H2O[t + 1, :], 0, 1)

    return C_CO2, C_H2O, C_CH4, C_H2, theta_CO2, theta_H2O,  P_grid, u_grid, rho_grid


