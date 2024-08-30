# CO2REAKT

**A Reaktoro framework for calculating carbon dioxide solubility in solutions containing salts and acid.**

## Overview

CO2REAKT is a comprehensive numerical framework developed to calculate CO₂ solubility in saline and acidic solutions. Built on the Reaktoro chemical simulator (https://github.com/reaktoro/reaktoro), this framework includes three solvers and a dataset essential for accurate solubility calculations.

## Datasets

### 1. Original Dataset
- **Source**: 15 experimental studies, with data extracted through table readings and digitization:
  
  1. **dos Santos, P.F., et al., 2022.** Experimental measurements and thermodynamic modeling of CO2 solubility in the NaCl-Na2SO4-CaCl2-MgCl2 system up to 150 °C and 200 bar. [DOI:10.2139/ssrn.4279657](https://doi.org/10.2139/ssrn.4279657)

  2. **Liu, B., et al., 2021.** Measurement of Solubility of CO2 in NaCl, CaCl2, MgCl2 and MgCl2 + CaCl2 Brines at Temperatures from 298 to 373 K and Pressures up to 20 MPa Using the Potentiometric Titration Method. Energies 14, 7222. [DOI:10.3390/en14217222](https://doi.org/10.3390/en14217222)

  3. **Lara Cruz, J., et al., 2021.** Experimental Study of Carbon Dioxide Solubility in Sodium Chloride and Calcium Chloride Brines at 333.15 and 453.15 K for Pressures up to 40 MPa. J. Chem. Eng. Data 66, 249–261. [DOI:10.1021/acs.jced.0c00592](https://doi.org/10.1021/acs.jced.0c00592)

  4. **Poulain, M., et al., 2019.** Experimental Measurements of Carbon Dioxide Solubility in Na–Ca–K–Cl Solutions at High Temperatures and Pressures up to 20 MPa. J. Chem. Eng. Data 64, 2497–2503. [DOI:10.1021/acs.jced.9b00023](https://doi.org/10.1021/acs.jced.9b00023)

  5. **Messabeb, H., et al., 2017.** Experimental Measurement of CO2 Solubility in Aqueous CaCl2 Solution at Temperature from 323.15 to 423.15 K and Pressure up to 20 MPa Using the Conductometric Titration. J. Chem. Eng. Data 62, 4228–4234. [DOI:10.1021/acs.jced.7b00591](https://doi.org/10.1021/acs.jced.7b00591)

  6. **Jacob, R., Saylor, B.Z., 2016.** CO2 solubility in multi-component brines containing NaCl, KCl, CaCl2 and MgCl2 at 297 K and 1–14 MPa. Chemical Geology 424, 86–95. [DOI:10.1016/j.chemgeo.2016.01.013](https://doi.org/10.1016/j.chemgeo.2016.01.013)

  7. **Gilbert, K., et al., 2016.** CO2 solubility in aqueous solutions containing Na+, Ca2+, Cl−, SO42− and HCO3-: The effects of electrostricted water and ion hydration thermodynamics. Applied Geochemistry 67, 59–67. [DOI:10.1016/j.apgeochem.2016.02.002](https://doi.org/10.1016/j.apgeochem.2016.02.002)

  8. **Zhao, H., et al., 2015.** Experimental studies and modeling of CO2 solubility in high temperature aqueous CaCl2, MgCl2, Na2SO4, and KCl solutions. AIChE Journal 61, 2286–2297. [DOI:10.1002/aic.14825](https://doi.org/10.1002/aic.14825)

  9. **Bastami, A., et al., 2014.** Experimental and modelling study of the solubility of CO2 in various CaCl2 solutions at different temperatures and pressures. Pet. Sci. 11, 569–577. [DOI:10.1007/s12182-014-0373-1](https://doi.org/10.1007/s12182-014-0373-1)

  10. **Tong, D., et al., 2013.** Solubility of CO2 in Aqueous Solutions of CaCl2 or MgCl2 and in a Synthetic Formation Brine at Temperatures up to 423 K and Pressures up to 40 MPa. J. Chem. Eng. Data 58, 2116–2124. [DOI:10.1021/je400396s](https://doi.org/10.1021/je400396s)

  11. **Liu, Y., et al., 2011.** Solubility of CO2 in aqueous solutions of NaCl, KCl, CaCl2 and their mixed salts at different temperatures and pressures. The Journal of Supercritical Fluids 56, 125–129. [DOI:10.1016/j.supflu.2010.12.003](https://doi.org/10.1016/j.supflu.2010.12.003)

  12. **Hamidi, H., et al., 2015.** Study of CO2 Solubility in Brine under Different Temperatures and Pressures. Advanced Materials Research 1113, 440–445. [DOI:10.4028/www.scientific.net/AMR.1113.440](https://doi.org/10.4028/www.scientific.net/AMR.1113.440)
 
  13. **Onda, K., et al., 1970.** Salting-Out Parameters of Gas Solubility in Aqueous Salt Solutions. Journal of Chemical Engineering of Japan 3, 18–24. [DOI:10.1252/jcej.3.18](https://doi.org/10.1252/jcej.3.18)

  14. **Yasunishi, A., Yoshida, F., 1979.** Solubility of carbon dioxide in aqueous electrolyte solutions. J. Chem. Eng. Data 24, 11–14. [DOI:10.1021/je60080a007](https://doi.org/10.1021/je60080a007)
 
  15. **Prutton, C.F., Savage, R.L., 1945.** The Solubility of Carbon Dioxide in Calcium Chloride-Water Solutions at 75, 100, 120° and High Pressures. J. Am. Chem. Soc. 67, 1550–1554. [DOI:10.1021/ja01225a047](https://doi.org/10.1021/ja01225a047)

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
- **Description**: A widely used model for calculating CO₂ solubilities in aqueous solutions containing salts (Duan, Z., Sun, R., 2003. An improved model calculating CO2 solubility in pure water and aqueous NaCl solutions from 273 to 533 K and from 0 to 2000 bar. Chemical Geology 193, 257–271. https://doi.org/10.1016/S0009-2541(02)00263-2).
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
