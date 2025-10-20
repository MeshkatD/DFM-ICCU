# DFM-ICCU: Open-Source Framework for Simulation and Optimisation of Dual-Function Material Processes

**Authors:**  
Meshkat Dolat¹, Andrew David Wright², Melis S. Duyar¹ ³, Michael Short¹,³*  

¹ School of Chemistry and Chemical Engineering, University of Surrey, UK  
² Department of Chemical Engineering, The University of Manchester, UK  
³ Institute for Sustainability, University of Surrey, UK  

---

## 📘 Overview
This repository provides the open-source Python models developed for the simulation and multi-objective optimisation of an **Integrated CO₂ Capture and Utilisation (ICCU)** process using **Dual-Function Materials (DFMs)**.  

The framework allows simulation of cyclic operation under **cyclic steady-state (CSS)** conditions and enables **algorithmic optimisation** to explore trade-offs between **methane purity**, **CO₂ recovery**, and **process productivity**.

Two complementary model sets are provided:

| Folder | Description |
|:-------|:-------------|
| **`simulation_x2/`** | Full 1D CSS reactor model including **mass**, **energy**, and **momentum** balances. Used to simulate the complete DFM cycle with detailed heat and pressure effects. |
| **`optimization_y/`** | Reduced (surrogate) model for efficient **multi-objective optimisation**, derived from the full model but decoupled from the heat balance for computational speed. |

---

## ⚙️ Features
- Comprehensive reactor model incorporating adsorption, purge, hydrogenation, and final purge stages.  
- Physically consistent cyclic steady-state simulation framework.  
- Multi-objective optimisation using the **NSGA-II** evolutionary algorithm via the **Optuna** platform.  
- Parameter bounds and KPI definitions consistent with the associated manuscript.  
- Open and reproducible implementation in Python.

---

## 🚀 How to Use

### 1. Install required Python packages
Create a virtual environment (optional) and install dependencies:
```bash
pip install numpy scipy pandas matplotlib optuna
```
### 2. Run the full CSS simulation
```bash
python simulation_x2/cycle_model_CSS_X2.py
```
3. Run the reduced optimisation model
```bash
python optimization_y/_cycle_model_CSS_Y.py
```

Each model will simulate cyclic operation and report the key performance indicators (methane purity, CO₂ recovery, and productivity).

📂 Repository Structure
```bash
DFM-ICCU/
│
├── simulation_x2/       # Full cyclic steady-state model (mass + energy + momentum)
│   ├── adsorption_model_X2.py
│   ├── purge_model_X2.py
│   ├── hydrogenation_model_X2.py
│   ├── cycle_model_CSS_X2.py
│   └── miscellaneous_X2.py
│
├── optimization_y/      # Reduced surrogate model for optimisation
│   ├── adsorption_model_Y.py
│   ├── purge_model_Y.py
│   ├── hydrogenation_model_Y.py
│   ├── _cycle_model_CSS_Y.py
│   └── miscellaneous_Y.py
│
└── README.md
```
📖 Related Publication

This repository accompanies the manuscript:
“Multi-objective Optimisation of an Integrated CO₂ Capture and Methanation Process Using Dual-Function Materials”
by Meshkat Dolat, Andrew D. Wright, Melis S. Duyar, and Michael Short (2025).

If you use this repository, please cite the above publication.

📄 License

This project is released under the MIT License, allowing free use and modification with attribution.

🤝 Acknowledgements

The authors gratefully acknowledge support from the University of Surrey, the University of Manchester, and the Institute for Sustainability.
This work was carried out as part of ongoing research in CO₂ capture, utilisation, and process optimisation using advanced catalytic materials.
