""" DFM CYCLE OPTIMISATION MODEL """

"""
Article:
Multi-objective optimisation for in CO2 utilisation process using Dual-Function Material

Authors:
Meshkat Dolat¹, Andrew David Wright², Melis S. Duyar¹,³, Michael Short¹,³*

Affiliations:
¹ School of Chemistry and Chemical Engineering, University of Surrey, UK
² Department of Chemical Engineering, School of Engineering, The University of Manchester, UK
³ Institute for Sustainability, University of Surrey, UK

"""

import numpy as np
import optuna
from optuna.samplers import BaseSampler
from optuna.trial import TrialState
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
from adsorption_model_Y import simulate_adsorption_model
from purge_model_Y import simulate_purge_model
from hydrogenation_model_Y import simulate_hydrogenation_model
import plotly.io as pio
from optuna.visualization import (
    plot_param_importances,
    plot_slice,
    plot_contour,
    plot_parallel_coordinate,
)
import warnings

# === choose which two objectives to optimise ===
ACTIVE_OBJECTIVES = ("recovery", "productivity")
# Options: ("recovery", "purity") or ("recovery", "productivity") or ("productivity", "purity") 


# =========================== PERFORMANCE METRICS ========================== #

def calculate_recovery(C_feed_ads_CO2, C_CH4_hyd, F_gas, Time_ads, Time_hyd,  T, P_ads, R_):
    """Calculate recovery using steady-state data."""
    # Calculate uniform time step sizes
    dt_hyd = Time_hyd / (len(C_CH4_hyd))                                            # Hydrogenation time step
    total_CH4_out = np.trapz(F_gas * C_CH4_hyd, dx=dt_hyd)                          # Total CH4 produced during hydrogenation
    total_CO2_in = F_gas * (P_ads * C_feed_ads_CO2/(R_ * T * 100)) * Time_ads       # Total CO2 fed during adsorption
    Recovery = total_CH4_out / total_CO2_in
    return Recovery


def calculate_productivity(C_CH4_hyd, F_gas, W_DFM, Time_cycle, Time_hyd):
    """Calculate productivity using steady-state data."""
    dt_hyd = Time_hyd / (len(C_CH4_hyd))                                            # Hydrogenation time step
    total_CH4_out = np.trapz(F_gas * C_CH4_hyd, dx=dt_hyd)                          # Total CH4 produced during hydrogenation
    Productivity = total_CH4_out / (W_DFM * Time_cycle)                             # Productivity (per kg DFM)
    return Productivity * 1000


def calculate_purity(C_CH4_hyd, C_H2_hyd, C_CO2_hyd, F_gas, Time_hyd):
    """Calculate purity using steady-state data."""
    dt_hyd = Time_hyd / (len(C_CH4_hyd))                                            # Hydrogenation time step
    total_CH4_out = np.trapz(F_gas * C_CH4_hyd, dx=dt_hyd)                          # Total CH4 produced during hydrogenation
    total_hydrogenation_out = np.trapz(F_gas * (C_CH4_hyd + C_H2_hyd + C_CO2_hyd), dx=dt_hyd)  # Total gas output (except water & N2)
    Purity = total_CH4_out / total_hydrogenation_out
    return Purity


def objective(trial):
    """
    Objective function for Optuna optimization, running until CSSC.

    Args:
        trial: Optuna trial object.

    Returns:
        tuple: A tuple containing recovery, productivity, and purity.
    """
    # capture SciPy RuntimeWarnings during the whole simulation
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always", category=RuntimeWarning)
        
        # ============= 1st Stage Variables =============
        Time_ads = trial.suggest_int('Time_ads', 50, 600)
        Time_prg = trial.suggest_int('Time_prg', 50, 200)
        Time_hyd = trial.suggest_int('Time_hyd', 50, 1000)
        Time_prg2 = trial.suggest_int('Time_prg2', 50, 200)
        F_gas = trial.suggest_int('F_gas', 5, 30)                                       # cm3/s
        Di = trial.suggest_float('Di', 0.8, 4.0, step=0.1)
        T_K = trial.suggest_int('T', (200 + 273), (400 + 273))                          # Temperature (K) (Assuming isothermal operation)
        
        # ============= 2nd Stage Variables =============
        W_DFM = 10.0                        # Total Weight of DFM in the reactor (gr)
        P_hyd = 1                           # Total pressure during hydrogenation (atm)
        C_feed_hyd_H2 = 10                  # H2  %vol in feed gas - Hydrogenation stage (mmol/cm^3)
        C_feed_hyd_N2 = 100-C_feed_hyd_H2   # N2  %vol in feed gas - Hydrogenation stage (mmol/cm^3)
        
        # ============= Other Parameters =============
        rho_bed = 0.5188                    # Adsorption bed Density (g/cm^3)
        V_react = W_DFM/rho_bed             # Reactor (Tube) Volue (mL)
        Ai = 3.14 * (Di**2) / 4             # Reactor (Tube) Internal Surface Area (cm2)
        L = V_react/Ai                      # Length of reactor bed (cm)
        u = F_gas / Ai                      # Linear velocity (cm/s)
        Omega = (4.10e-06) * (T_K - 273)**2 + (-4.60e-03) * (T_K - 273) + (1.499)       # Maximum adsorption capacity (mmol/g)
        D = 0.16                            # Diffusion coefficient (cm^2/s)
        epsilon = 0.35                      # Porosity 
        P_ads = 1                           # Total pressure during adsorption (atm)
        P_prg = 1                           # Total pressure during purge (atm)
    
        # Catalyst parameters
        rho_s = 3.98  # (g/cm3)             # DFM density (mostly Alumina oxide)
        CP_s = 0.955  # (J/g.K)             # based on heat capacity of alumina (Al₂O₃)
        dp = 0.1                            # cm, particle diameter (1 mm)
    
        # constants
        R_gas   = 8.314                     # J/(mol·K)
        R_      = 0.08206                   # (L.atm / mol.K)
    
        # Model Params
        Nx = 10                              # Number of spatial grid points
        Nt_ads = 1000                        # Number of time steps
        Nt_prg = 1000                        # Number of time steps
        Nt_hyd = 3000                        # Number of time steps
        Nt_prg2 = 1000                       # Number of time steps
    
        C_feed_ads_CO2 = 12                 # CO2 %vol in feed gas - Adsorption stage (mmol/cm^3)  
        C_feed_ads_H2O = 0                  # H2O %vol in feed gas - Adsorption stage (mmol/cm^3)
        C_feed_prg_CO2 = 0                  # CO2 %vol in feed gas - Purge stage (mmol/cm^3)
        C_feed_prg_H2O = 0                  # H2O %vol in feed gas - Purge stage (mmol/cm^3)
    
        # Kinetic Params 
        E6 = 34                     # Activation energy for desorption of CO2 from the adsorption sites during purge stage(J/mmol)          **(E4) in new model
        Alfa = 0.5                  # Adsorption strength correction factor
        E7 = 14                     # Activation energy for desorption of CO2 from the adsorption sites during hydrogenation stage(J/mmol)  **(E6) in new model
        E8 = 65                     # Activation energy for formation of CH4 during Hydrogenation stage (J/mmol)                            **(E5) in new model
        E10 = 10                    # Activation energy for adsorption of H2O on the adsorption sites during Hydrogenation stage (J/mmol)   **(E7) in new model
    
        k1 = 60                     # Optimized kinetic constant for ads. of CO2 on CO2 sites (cm3/s.g)
        k2 = 20                     # Optimized kinetic constant for ads. of H2O on CO2 sites (cm3/s.g)
        k3 = 1                      # Optimized kinetic constant for ads. of H2O on H2O sites (cm3/s.g)
        k4 = 75.1                   # Optimized kinetic constant for ads. of H2O on H2O sites (cm3/s.g)                                     **X Deleted in new model
        k5 = 19.8                   # Optimized kinetic constant for ads. of H2O on H2O sites (cm3/s.g)                                     **X Deleted in new model
        k6 = 0.012                  # Optimized pre-exponential constant for purge of CO2 from CO2 sites (mmol/g.s)                         **(k4) in new model
        k7 = 38                     # Optimized kinetic constant for desorption of CO2 from CO2 sites during hydrogenation (cm3/s.g)        **(k6) in new model
        k8 = 43960                  # Optimized kinetic constant for CO2 hydrogenation (mmol/g.s.atm^5n)                                    **(k5) in new model
        k9 = 0.003                  # Optimized kinetic constant for des. of H2O from H2O sites during hydrogenation(mmol/g.s)              **(k8) in new model
        k10 = 144                   # Optimized kinetic constant for ads. of H2O on H2O sites (cm3/s.g)                                     **(k7) in new model
    
        K_CO2 = 375.3               # Optimized adsorption constant of CO2 (cm3/mmol)                                                       **X Deleted in new model
        m = 1                       # Adjust parameter in H2O adsorption                                                                    **X Deleted in new model
        n = 0.14                    # Adjust parameter in CH4 production
    
        # Analyser Params (to replicate delay response and smothening effect)
        tau_increase = 20            # Time constant for increasing step (Analyser delay) 19.2
        tau_decrease = 20            # Time constant for decreasing step (Analyser delay) 22
        
        total_cycle_time = Time_ads + Time_prg + Time_hyd + Time_prg2
    
        # Initialize for CSSC loop
        C_CO2, C_H2O, theta_CO2, theta_H2O = np.zeros(Nx), np.zeros(Nx), np.zeros(Nx), np.zeros(Nx)
        C_CO2_init_ads, C_H2O_init_ads = np.zeros(Nx), np.zeros(Nx)             #initial concentration of CO2 and H2O inside the column (before running the cycle)
        theta_CO2_init_ads, theta_H2O_init_ads = np.zeros(Nx), np.zeros(Nx)     #initial coverage area for CO2 and H2O inside the column (before running the cycle)
        
        # --- Begin CSSC loop --- #
        cycles_to_converge = 5     # Default to the maximum number of cycles
        converged = False           # Track if convergence occurs
        
        for cycle in range(5):
            print(f"Cycle {cycle+1}")
    
            # Adsorption
            C_CO2_ads, C_H2O_ads, theta_CO2_ads, theta_H2O_ads, P_grid_ads, u_grid_ads, rho_grid_ads = simulate_adsorption_model(
                k1, k2, k3, k4, k5, K_CO2, m, T_K, P_ads, D, L, Time_ads, u, epsilon,
                C_feed_ads_CO2, C_feed_ads_H2O, C_CO2_init_ads, C_H2O_init_ads, theta_CO2_init_ads, theta_H2O_init_ads,
                rho_bed, Omega, Nx, Nt_ads
            )
            
            # Purge
            C_CO2_prg, C_H2O_prg, C_CH4_prg, C_H2_prg, theta_CO2_prg, theta_H2O_prg, P_grid_prg, u_grid_prg, rho_grid_prg= simulate_purge_model(
                k2, k3, k4, k5, k6, K_CO2, m, E6, T_K, P_prg, Alfa, D, L, Time_prg, u, epsilon,
                C_feed_prg_CO2, C_feed_prg_H2O, 
                C_CO2_ads[-1, :], C_H2O_ads[-1, :], 0, 0, theta_H2O_ads[-1, :], theta_CO2_ads[-1, :], 
                rho_bed, Omega, Nx, Nt_prg
            )
    
            # Hydrogenation
            C_CO2_hyd, C_CH4_hyd, C_H2_hyd, C_H2O_hyd, theta_CO2_hyd, theta_H2O_hyd, P_grid_hyd, u_grid_hyd, rho_grid_hyd= simulate_hydrogenation_model(
                k7, k8, k9, k10, E7, E8, E10, T_K, P_hyd, n, D, L, Time_hyd, u, epsilon,
                C_feed_hyd_H2, C_feed_hyd_N2,
                C_CO2_prg[-1, :], C_H2O_prg[-1, :], theta_CO2_prg[-1, :], theta_H2O_prg[-1, :], 
                rho_bed, Omega, Nx, Nt_hyd
            )
    
            # Final Purge
            C_CO2_prg2, C_H2O_prg2, C_CH4_prg2, C_H2_prg2, theta_CO2_prg2, theta_H2O_prg2, P_grid_prg2, u_grid_prg2, rho_grid_prg2 = simulate_purge_model(
                k2, k3, k4, k5, k6, K_CO2, m, E6, T_K, P_prg, Alfa, D, L, Time_prg2, u, epsilon,
                C_feed_prg_CO2, C_feed_prg_H2O, 
                C_CO2_hyd[-1, :], C_H2O_hyd[-1, :], C_CH4_hyd[-1, :], C_H2_hyd[-1, :], theta_H2O_hyd[-1, :], theta_CO2_hyd[-1, :], 
                rho_bed, Omega, Nx, Nt_prg2
            )
                   
            # Convergence check
            if (
                np.all(np.abs(C_CO2_prg2[-1, :] - C_CO2) < 1e-3) and
                np.all(np.abs(C_H2O_prg2[-1, :] - C_H2O) < 1e-3) and
                np.all(np.abs(theta_CO2_prg2[-1, :] - theta_CO2) < 0.05) and
                np.all(np.abs(theta_H2O_prg2[-1, :] - theta_H2O) < 0.05)
            ):
                cycles_to_converge = cycle + 1
                converged = True
                break   # Exit the Loop if converged
    
            # Update for next cycle
            C_CO2, C_H2O = C_CO2_prg2[-1, :], C_H2O_prg2[-1, :]
            theta_CO2, theta_H2O = theta_CO2_prg2[-1, :], theta_H2O_prg2[-1, :]
            C_CO2_init_ads, C_H2O_init_ads = C_CO2_prg2[-1, :], C_H2O_prg2[-1, :]
            theta_CO2_init_ads, theta_H2O_init_ads = theta_CO2_prg2[-1, :], theta_H2O_prg2[-1, :]
        
        if not converged:
            # Return a penalty value for non-converged trials
            pass
    
        # Calculate metrics using the steady-state cycle data
        
        Recovery = calculate_recovery(C_feed_ads_CO2, C_CH4_hyd[:, -1], F_gas, Time_ads, Time_hyd, T_K, P_ads, R_)
        Productivity = calculate_productivity(C_CH4_hyd[:, -1], F_gas, W_DFM, total_cycle_time, Time_hyd)
        Purity = calculate_purity(C_CH4_hyd[:, -1], C_H2_hyd[:, -1], C_CO2_hyd[:, -1], F_gas, Time_hyd)
        
        metrics = {"recovery": Recovery, "productivity": Productivity, "purity": Purity}
    
    # after the run, summarise warnings for this trial
    warn_msgs = [str(w.message) for w in wlist if issubclass(w.category, RuntimeWarning)]
    warn_flag = len(warn_msgs) > 0
    
    # store on the trial so we can export later
    trial.set_user_attr("warn_flag", warn_flag)
    trial.set_user_attr("warn_count", len(warn_msgs))
    # keep a short sample to avoid huge cells
    if warn_flag:
        trial.set_user_attr("warn_sample", warn_msgs[0][:160])

    
    return tuple(metrics[name] for name in ACTIVE_OBJECTIVES)



# Defining a custom sampler in Optuna that prioritises the lower and upper bounds of search space
class BoundaryPrioritizedSampler(BaseSampler):
    def __init__(self, base_sampler=None):
        self.base_sampler = base_sampler or optuna.samplers.TPESampler()
        self.bound_priority_list = []  # Track which bounds to prioritize next

    def infer_relative_search_space(self, study, trial):
        return self.base_sampler.infer_relative_search_space(study, trial)

    def sample_relative(self, study, trial, search_space):
        # Default to base sampler for relative search space sampling
        return self.base_sampler.sample_relative(study, trial, search_space)

    def sample_independent(self, study, trial, param_name, param_distribution):
        # If bounds are not yet exhausted, prioritize bounds
        if param_name not in self.bound_priority_list:
            self.bound_priority_list.append(param_name)
            return param_distribution.low  # Prioritize lower bound first
        elif self.bound_priority_list.count(param_name) == 1:
            self.bound_priority_list.append(param_name)
            return param_distribution.high  # Then prioritize upper bound
        else:
            # Default to base sampler once bounds are sampled
            return self.base_sampler.sample_independent(study, trial, param_name, param_distribution)

   
if __name__ == "__main__":
    start_time = perf_counter()

    # Create Optuna study with multi-objective optimization
    study = optuna.create_study(
        directions=["maximize"] * len(ACTIVE_OBJECTIVES),
        sampler=optuna.samplers.NSGAIISampler()
    )

    # ---- run optimisation ----
    study.optimize(objective, n_trials=10)

    # ---- collate results (all trials) ----
    obj_names = list(ACTIVE_OBJECTIVES)
    all_points = []
    for t in study.trials:
        if t.state == TrialState.COMPLETE:
            row = {obj_names[i]: t.values[i] for i in range(len(obj_names))}
            row.update(t.params)
            # NEW: warning flags
            row["warn_flag"]  = bool(t.user_attrs.get("warn_flag", False))
            row["warn_count"] = int(t.user_attrs.get("warn_count", 0))
            row["warn_sample"] = t.user_attrs.get("warn_sample", "")
            all_points.append(row)

    # ---- extract Pareto front (best_trials) ----
    pareto_points = []
    for t in study.best_trials:
        row = {obj_names[i]: t.values[i] for i in range(len(obj_names))}
        row.update(t.params)
        row["warn_flag"]  = bool(t.user_attrs.get("warn_flag", False))
        row["warn_count"] = int(t.user_attrs.get("warn_count", 0))
        row["warn_sample"] = t.user_attrs.get("warn_sample", "")
        pareto_points.append(row)

    # ---- save to Excel ----
    try:
        pd.DataFrame(all_points).to_excel(f"all_points_{'-'.join(obj_names)}.xlsx", index=False)
        pd.DataFrame(pareto_points).to_excel(f"pareto_points_{'-'.join(obj_names)}.xlsx", index=False)
    except Exception as e:
        print("Warning: could not write Excel files:", e)

    # ---- 2D Pareto scatter ----
    if len(obj_names) == 2 and len(all_points) > 0:
        x_name, y_name = obj_names[0], obj_names[1]
        try:
            plt.figure(figsize=(6, 5))
            plt.scatter([p[x_name] for p in all_points],
                        [p[y_name] for p in all_points],
                        s=20, alpha=0.4, label="all trials")
            if len(pareto_points) > 0:
                plt.scatter([p[x_name] for p in pareto_points],
                            [p[y_name] for p in pareto_points],
                            s=40, label="Pareto")
            plt.xlabel(x_name.capitalize())
            plt.ylabel(y_name.capitalize())
            plt.title(f"Pareto front: {x_name} vs {y_name}")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Warning: could not render Pareto scatter:", e)


    # ---- Plotly: set renderer and also write HTML fallbacks ----
    try:
        import plotly.io as pio
        # Force browser rendering; change to "chrome", "firefox", etc. if needed.
        pio.renderers.default = "browser"
    except Exception as e:
        print("Warning: could not set Plotly renderer:", e)

    # ---- Hyperparameter importances for each active objective ----
    try:
        for i, name in enumerate(obj_names):
            fig = plot_param_importances(
                study,
                target=lambda t, ii=i: t.values[ii],  # pick objective i
                target_name=name
            )
            fig.update_layout(title=f"Hyperparameter importances for {name}")
            # Try to show, and also save an HTML copy
            try:
                fig.show()
            finally:
                out_html = f"imp_{name}_{'-'.join(obj_names)}.html"
                fig.write_html(out_html, include_plotlyjs="cdn")
                print(f"[saved] {out_html}")
    except Exception as e:
        print("Warning: could not render importance plots:", e)

    # ---- Slice / Contour / Parallel plots (per-objective) ----
    try:
        from optuna.visualization import plot_slice, plot_contour, plot_parallel_coordinate
        # Get parameter list from a completed trial
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        param_names = list(completed[0].params.keys()) if completed else []

        if param_names:
            for i, name in enumerate(obj_names):
                # SLICE
                fig_s = plot_slice(
                    study,
                    params=param_names,
                    target=lambda t, ii=i: t.values[ii],
                    target_name=name
                )
                fig_s.update_layout(title=f"Slice plots for {name}")
                try:
                    fig_s.show()
                finally:
                    out_html = f"slice_{name}_{'-'.join(obj_names)}.html"
                    fig_s.write_html(out_html, include_plotlyjs="cdn")
                    print(f"[saved] {out_html}")

                # CONTOUR (pairwise interactions)
                fig_c = plot_contour(
                    study,
                    params=param_names,
                    target=lambda t, ii=i: t.values[ii],
                    target_name=name
                )
                fig_c.update_layout(title=f"Contour plots for {name}")
                try:
                    fig_c.show()
                finally:
                    out_html = f"contour_{name}_{'-'.join(obj_names)}.html"
                    fig_c.write_html(out_html, include_plotlyjs="cdn")
                    print(f"[saved] {out_html}")

                # PARALLEL COORDINATE
                fig_p = plot_parallel_coordinate(
                    study,
                    params=param_names,
                    target=lambda t, ii=i: t.values[ii],
                    target_name=name
                )
                fig_p.update_layout(title=f"Parallel coordinates for {name}")
                try:
                    fig_p.show()
                finally:
                    out_html = f"parallel_{name}_{'-'.join(obj_names)}.html"
                    fig_p.write_html(out_html, include_plotlyjs="cdn")
                    print(f"[saved] {out_html}")
        else:
            print("No completed trials with parameters found for slice/contour/parallel plots.")
    except Exception as e:
        print("Warning: could not render slice/contour/parallel plots:", e)

    elapsed = perf_counter() - start_time
    print(f"Done. Elapsed time: {elapsed/60:,.2f} min")
