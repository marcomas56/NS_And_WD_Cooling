# Neutron Star and White Dwarf Cooling Simulation

This project simulates the thermal evolution (cooling) of a Neutron Star or a White Dwarf over time. It solves the heat diffusion equation using the **Crank-Nicolson** numerical method.

## 👥 Authors
* **Yeray Antón**
* **Marco Mas**

## 🔭 Description
The simulation models the cooling process by considering detailed microphysics:
1. **Neutron Star**:
  * **Specific Heat ($C_v$):** Contributions from neutrons, protons, electrons, and ions.
  * **Neutrino Emissivity ($Q$):** Energy loss via Urca processes, Bremsstrahlung...
  * **Stellar Structure:** Uses density profiles for 1.1, 1.4, and 1.7 Solar Masses.
2. **White Dwarf**:
  * **Specific Heat ($C_v$):** Contributions from electron.
  * **Emissivity ($Q$):** Energy loss via e-N Bremsstrahlung and black-body radiation.
  * **Stellar Structure:** Uses the Fermi gas model.

## 📋 Requirements
You need Python installed along with these libraries:

```bash
pip install numpy matplotlib scipy
```


🚀 How to Run
Run the main script in your terminal:


```bash
python Neutron_Star_Cooling\Cooling.py
python White_Dwarf_Cooling\Cooling.py
```

### Interactive Inputs
Upon execution, the program will request two inputs via the console:

1.  **Select Star Mass:** Choose the stellar model to simulate (Only in the Neutron Star code).
    * `1`: 1.1 Solar Masses (File: `PL200-1.10.DAT`)
    * `2`: 1.4 Solar Masses (File: `PL200-1.40.DAT`)
    * `3`: 1.7 Solar Masses (File: `PL200-1.70.DAT`)

2.  **Save Animation:** Decide if you want to generate a GIF file.
    * `1`: **Yes** (Saves `Temperatura.gif`).
    * `2`: **No** (Only displays the plots).

## 📂 Project Structure
1. 📂 **Neutron_Star_Cooling**:
  * `Cooling.py`: **Main script.** Contains the time-evolution loop, the Crank-Nicolson solver, adaptive time-stepping logic, and plotting functions.
  * `CV.py`: Module for calculating the total **specific heat** ($C_v$).
  * `Emisividad.py`: Module for calculating the total **emissivity** ($Q_{total}$).
  * `PL200-*.DAT`: Data files containing the density profiles and structural composition of the neutron stars.
2. 📂 **White_Dwarf_Cooling**:
  * `Cooling.py`: **Main script.** Contains the time-evolution loop, the Crank-Nicolson solver, adaptive time-stepping logic, and plotting functions.
  * `Emisividad.py`: Module for calculating the total **emissivity** ($Q_{total}$).
