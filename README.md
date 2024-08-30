# CO2REAKT

**A Reaktoro framework for calculating carbon dioxide solubility in solutions containing salts and acid**

## Overview

CO2REAKT is a comprehensive numerical framework developed to calculate CO₂ solubility in saline and acidic solutions. Built on the Reaktoro chemical simulator (https://github.com/reaktoro/reaktoro), this framework includes three solvers and two datasets essential for accurate solubility calculations.

## Datasets

### 1. Original Dataset
- **Source**: 15 experimental studies, with data extracted through table readings and digitization.
- **Total Data Points**: 1,094
- **Variables**: Pressure, Temperature, CaCl₂, MgCl₂, NaCl, KCl concentrations, and CO₂ solubility.
- **Units**: As originally extracted from the source studies.

### 2. Preprocessed Dataset
- **Data Points**: 999 (after preprocessing).
- **Preprocessing Steps**:
  - Removed outliers and redundant data.
  - Excluded data points with pressures or solubilities equal to zero.
  - Standardized units to MPa (Pressure), Kelvin (Temperature), and mol/kg (Concentrations).

## Solvers

### 1. Duan and Sun (2003) Model Solver
- **Description**: A widely used model for calculating CO₂ solubilities in aqueous solutions containing salts.
- **Inputs**: Calcium, Sodium, Potassium, Magnesium cations, Chlorine, and Sulfate ions, along with system pressure and temperature.
- **Usage**: Reads from the preprocessed dataset and calculates CO₂ solubility.

### 2. CO2REAKT Framework
- **Description**: A developed framework based on the Reaktoro chemical simulator.
- **Inputs**: Pressure, Temperature, and molalities of salts from the preprocessed dataset.
- **Databases**: Utilizes Reaktoro's built-in databases such as AQ17, SUPCRTBL, and PHREEQC.
- **Aqueous Species Activity Models**: For calculating aqueous species activities, excluding CO₂, activity models such as HKF, Davies, and Debye-Huckel were used. For calculating aqueous CO₂ activity, the model of Duan and Sun is used.
- **Gaseous Species Activity Models**: Duan and Sun activity model for aqueous CO₂ and Spycher-Pruess-Ennis for gaseous CO₂ and water.
- **Functionality**: Predicts CO₂ solubility using Gibbs Energy Minimization (GEM) with accuracy similar to the Duan and Sun model.

### 3. Extended CO2REAKT Framework
- **Description**: An extension of the CO2REAKT framework designed to account for HCl concentration in solutions containing salts.
- **Inputs**: Similar to the original framework, with additional consideration of HCl concentration.
- **Activity Models**: Uses general activity models such as HKF, Davies, and Debye-Huckel for all aqueous species, including CO₂.
- **Functionality**: Though less accurate than the original framework, it effectively accounts for the impact of HCl on CO₂ solubility, highlighting the significant errors (up to 1063%) that can occur if HCl is not considered.

## Usage

1. **Install Reaktoro**:
- Use instructions to install Reaktoro (https://reaktoro.org/installation).

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/mcrossover97/CO2REAKT.git

3. **Access Datasets**:
- The original and preprocessed datasets are available in the `datasets/` directory.
- Load and explore the datasets using the provided data exploration scripts or your own analysis tools.

4. **Run Solvers**:
- Navigate to the `solvers/` directory.
- Run the solver scripts using your preferred Python environment.

## License

This project is licensed under the GNU Lesser General Public License v2.1 License. See the `LICENSE` file for more details.

## Contact

If you have any questions, feel free to email [khojastehmehr.mohammad@gmail.com](mailto:khojatehmehr.mohammad@gmail.com).
