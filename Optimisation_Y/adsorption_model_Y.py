
import numpy as np
from scipy.optimize import fsolve

def simulate_adsorption_model(
    k1, k2, k3, k4, k5, K_CO2, m,
    T, P_ads, D, L, Time_ads, u, epsilon,
    C_feed_ads_CO2, C_feed_ads_H2O,
    C_CO2_init_ads, C_H2O_init_ads,
    theta_CO2_init_ads, theta_H2O_init_ads,
    rho, Omega, Nx, Nt
):
    dx = L / (Nx - 1)
    dt = Time_ads / Nt

    # allocate solution arrays: time x space
    C_CO2       = np.zeros((Nt+1, Nx))
    C_H2O       = np.zeros((Nt+1, Nx))
    theta_CO2   = np.zeros((Nt+1, Nx))
    theta_H2O   = np.zeros((Nt+1, Nx))

    # constants
    R_gas   = 8.314             # J/(mol·K)
    M_mix   = 0.028             # kg/mol, approximate
    mu      = 3.0e-5            # Pa·s, N2 viscosity (350C)
    dp      = 1e-3              # m, particle diameter (1 mm)

    # convert inlet pressure from atm to Pa
    P0 = P_ads * 101325.0       # Pa

    # inlet concentration [mmol/mL] via ideal-gas at inlet pressure
    C0_CO2 = (C_feed_ads_CO2/100.0) * P0/(R_gas*T) * 1e-3
    C0_H2O = (C_feed_ads_H2O/100.0) * P0/(R_gas*T) * 1e-3

    # initial profile (t=0)
    C_CO2[0,:]       = C_CO2_init_ads
    C_H2O[0,:]       = C_H2O_init_ads
    theta_CO2[0,:]   = theta_CO2_init_ads
    theta_H2O[0,:]   = theta_H2O_init_ads

    # precompute Ergun pressure drop and local velocity along bed
    P_grid = np.zeros(Nx)
    u_grid = np.zeros(Nx)               # cm/s
    rho_grid = np.zeros(Nx)
    P_grid[0] = P0
    u_grid[0] = u
    rho_grid[0] = (P_grid[0]/(R_gas*T)) * M_mix

    for j in range(Nx-1):
        rho_mol = P_grid[j]/(R_gas*T)       # mol/m3
        rho_fl  = rho_mol * M_mix       # kg/m3
        rho_grid[j] = rho_fl
        u_m_s = u_grid[j] / 100.0       # cm/s to m/s
        
        term1   = 150*mu*(1-epsilon)**2/(epsilon**3*dp**2)*u_m_s
        term2   = 1.75*rho_fl*(u_m_s**2)*(1-epsilon)/(epsilon**3*dp)
        
        dPdz    = -(term1 + term2)      # Pa/m
        P_grid[j+1] = P_grid[j] + dPdz * (dx/100)
        # ideal gas mass continuity: u * P = const
        u_grid[j+1] = u * (P0 / P_grid[j+1])
    
    rho_grid[-1] = (P_grid[-1]/(R_gas*T)) * M_mix

    # residuals for implicit step at (t,i)
    def backward_euler_equations(y, i, t):
        C_CO2_next, C_H2O_next, th_CO2_next, th_H2O_next = y

        # diffusion (central / one-sided at boundaries)
        if i==1:
            diff_CO2 = (D*dt/epsilon)*(C_CO2[t,i+1] - C_CO2_next)/dx**2
            diff_H2O = (D*dt/epsilon)*(C_H2O[t,i+1] - C_H2O_next)/dx**2
        elif i==Nx-1:
            diff_CO2 = (D*dt/epsilon)*(-C_CO2_next + C_CO2[t+1,i-1])/dx**2
            diff_H2O = (D*dt/epsilon)*(-C_H2O_next + C_H2O[t+1,i-1])/dx**2
        else:
            diff_CO2 = (D*dt/epsilon)*(C_CO2[t,i+1] - 2*C_CO2_next + C_CO2[t+1,i-1])/dx**2
            diff_H2O = (D*dt/epsilon)*(C_H2O[t,i+1] - 2*C_H2O_next + C_H2O[t+1,i-1])/dx**2

        # convection
        u_loc     = u_grid[i]
        conv_CO2  = -u_loc*dt/epsilon * (C_CO2_next - C_CO2[t+1,i-1]) / dx
        conv_H2O  = -u_loc*dt/epsilon * (C_H2O_next - C_H2O[t+1,i-1]) / dx

        # scale concentrations by local pressure (partial pressure effect)
        C2eff = C_CO2_next * (P_grid[i]/P0)
        Heff = C_H2O_next * (P_grid[i]/P0)

        # reaction rates (in mmol/mL/s)
        r_CO2 = -k1*C2eff*(1 - th_CO2_next - th_H2O_next) - k2*C2eff*th_H2O_next
        r_H2O = (k2*C2eff*th_H2O_next - k3*Heff*(1 - th_CO2_next - th_H2O_next))

        # residual = new - old - dt*(diff+conv+reaction)
        eq1 = C_CO2_next - C_CO2[t,i] - diff_CO2 - conv_CO2 - (rho*r_CO2*dt/epsilon)
        eq2 = C_H2O_next - C_H2O[t,i] - diff_H2O - conv_H2O - (rho*r_H2O*dt/epsilon)
        eq3 = th_CO2_next - theta_CO2[t,i] + r_CO2*dt/Omega
        eq4 = th_H2O_next - theta_H2O[t,i] + r_H2O*dt/Omega
        return [eq1, eq2, eq3, eq4]

    # time marching
    for t in range(Nt):
        # inlet boundary at next time
        C_CO2[t+1,0]       = C0_CO2   # to be corrected! linked to the inlet pressure
        C_H2O[t+1,0]       = C0_H2O   # to be corrected! linked to the inlet pressure

        # solve interior nodes
        for i in range(1, Nx):
            guess = [C_CO2[t,i], C_H2O[t,i], theta_CO2[t,i], theta_H2O[t,i]]
            sol   = fsolve(backward_euler_equations, guess, args=(i,t))
            C_CO2[t+1,i], C_H2O[t+1,i], theta_CO2[t+1,i], theta_H2O[t+1,i] = sol

        # enforce physical bounds on coverages
        theta_CO2[t+1,:]   = np.clip(theta_CO2[t+1,:],   0, 1)
        theta_H2O[t+1,:]   = np.clip(theta_H2O[t+1,:],   0, 1)

    return C_CO2, C_H2O, theta_CO2, theta_H2O, P_grid, u_grid, rho_grid

