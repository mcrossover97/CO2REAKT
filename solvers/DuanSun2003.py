#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data preparation 
import pandas as pd
import numpy as np

# Molecular weights of the salts
molecular_weights = {
    'CaCl2': 0.11098,
    'NaCl': 0.05844,
    'KCl': 0.07455,
    'MgCl2': 0.09521,
}

# Molecular weight of water
mw_water = 0.018015  # kg/mol

# Molecular weight of CO2
mw_co2 = 0.04401  # kg/mol

# Conversion function for weight percentage to mol/kg considering all salts
def convert_weight_percent_to_mol_kg(value, mw, salt_weights):
    salt_weight_sum = sum(salt_weights.values)
    return value / (mw * (100 - salt_weight_sum))

# Conversion function for ppm to mol/kg considering all salts
def convert_ppm_to_mol_kg(value, mw, salt_weights):
    salt_weight_sum = sum(salt_weights.values)
    return value / (mw * (1_000_000 - salt_weight_sum))

# Conversion function for g/kg to mol/kg considering all salts
def convert_g_kg_to_mol_kg(value, mw):
    return value / (mw * 1000)


# Function to convert concentrations based on unit
def convert_concentration(row, col_name, unit, original_salt_weights):
    value = row[col_name]
    mw = molecular_weights[col_name.split()[0]]
    
    # Get the original salt weights from the copy
    salt_weights = original_salt_weights.loc[row.name]
    
    if unit == 'ppm':
        return convert_ppm_to_mol_kg(value, mw, salt_weights)
    elif unit == 'wt%':
        return convert_weight_percent_to_mol_kg(value, mw, salt_weights)
    elif unit == 'g/kg':
        return convert_g_kg_to_mol_kg(value, mw)
    else:
        return value

# Revised function to convert solubility based on unit
def convert_solubility(row, solubility_col, solubility_unit, salt_columns):
    value = row[solubility_col]
    unit = row[solubility_unit]
    
    # Get the converted salt concentrations in mol/kg
    salt_concentrations_mol_kg = {salt: row[salt] for salt in salt_columns}
    
    # Convert salt concentrations from mol/kg to kg
    salt_concentrations_kg = {salt: conc * molecular_weights[salt.split()[0]] for salt, conc in salt_concentrations_mol_kg.items()}
    
    # Calculate total weight including salts
    total_weight = sum(salt_concentrations_kg.values()) + (100 - sum(salt_concentrations_mol_kg.values())) * mw_water
    
    if unit == 'wt%':
        # Convert CO2 wt% to mol/kg considering all salts 
        if value == 0:
            molality_CO2 = 0
        else:
            molality_CO2 = value / (100 - value) / mw_co2
        return molality_CO2
    elif unit == 'molefraction':
        # Calculate moles of CO2
        moles_CO2 = value / (1.0 - value) / mw_water
        return moles_CO2
    else:
        return value

    
# Conversion function for temperature
def convert_temperature(value, unit):
    if unit == 'celsius':
        return value + 273.15  # Convert Celsius to Kelvin
    else:
        return value

# Conversion function for pressure
def convert_pressure(value, unit):
    if unit == 'atm':
        return value * 0.101325  # Convert atm to MPa
    if unit =='bar':
        return value * 0.1  # Convert atm to MPa
    else:
        return value

# Main function to read and process the Excel file
def read_and_rename_columns(file_path):
    # Define the actual column names as they appear in the Excel file
    columns_to_read = ['Paper Title', 'Temperature', 'Temperature Unit', 'Pressure', 'Pressure Unit','CaCl2 Concentration', 'NaCl Concentration', 'KCl Concentration','MgCl2 Concentration', 'Concentration Unit', 'CO2 Solubility', 'Solubility Unit']



    salt_columns = ['CaCl2 Concentration', 'NaCl Concentration', 'KCl Concentration', 'MgCl2 Concentration']
    
    # Read the Excel file
    df = pd.read_csv(file_path ,usecols=columns_to_read)

    # Make a copy of the original salt weights
    original_salt_weights = df[salt_columns].copy()
    
    # Convert temperature to Kelvin if needed
    df['Temperature'] = df.apply(lambda row: convert_temperature(row['Temperature'], row['Temperature Unit']), axis=1)
    df['Temperature Unit'] = 'kelvin'
    
  
    # Convert pressure to MPa if needed
    df['Pressure'] = df.apply(lambda row: convert_pressure(row['Pressure'], row['Pressure Unit']), axis=1)
    df['Pressure Unit'] = 'MPa'
    
    # Convert concentrations to mol/kg if needed
   # if df['Concentration Unit'].iloc[0] != 'mol/kg':
    for salt in salt_columns:
        df[salt] = df.apply(lambda row: convert_concentration(row, salt, row['Concentration Unit'], original_salt_weights), axis=1)
    df.loc[df['Concentration Unit']!='mol/l','Concentration Unit'] = 'mol/kg'

    # Convert solubility to mol/kg if needed
    #if df['Solubility Unit'].iloc[0] != 'mol/kg':
    df['CO2 Solubility'] = df.apply(lambda row: convert_solubility(row, 'CO2 Solubility', 'Solubility Unit', salt_columns), axis=1)
    df['Solubility Unit'] = 'mol/kg'
    
    return df


# In[ ]:


# Duanusn method for CO2 Solubility
import numpy as np
from scipy.optimize import newton

def calculate_co2_solubility_Duansun(df):
    df=df.copy()
    def get_co2_solubility_Duansun(row):

        def calculate_P_H2O(T):
            Tc = 647.29  # Critical temperature in K
            Pc = 220.85  # Critical pressure in bar
            c1 = -38.640844
            c2 = 5.8948420
            c3 = 59.876516
            c4 = 26.654627
            c5 = 10.637097
            t = (T - Tc) / Tc
            P_H2O = (Pc * T / Tc) * (1 + c1 * (-t)**1.9 + c2 * t + c3 * t**2 + c4 * t**3 + c5 * t**4)
            return P_H2O
        
        def calculate_y_CO2(P, P_H2O):
            return (P - P_H2O) / P
        
        def calculate_Pr(P):
            #Pc = 647.29  # Critical pressure in bar
            Pc = 73.8  # Critical pressure in bar
            return P / Pc
        
        def calculate_Tr(T):
            #Tc = 220.85  # Critical temperature in K
            Tc = 304.15  # Critical temperature in K
            return T / Tc
        
        def equation_Vr(Vr, Pr, Tr):
            a1 = 8.99288497e-2
            a2 = -4.94783127e-1
            a3 = 4.77922245e-2
            a4 = 1.03808883e-2
            a5 = -2.82516861e-2
            a6 = 9.49887563e-2
            a7 = 5.20600880e-4
            a8 = -2.93540971e-4
            a9 = -1.77265112e-3
            a10 = -2.51101973e-5
            a11 = 8.93353441e-5
            a12 = 7.88998563e-5
            a13 = -1.66727022e-2
            a14 = 1.39800000
            a15 = 2.96000000e-2
            term1 = (a1 + a2/Tr**2 + a3/Tr**3) / Vr
            term2 = (a4 + a5/Tr**2 + a6/Tr**3) / Vr**2
            term3 = (a7 + a8/Tr**2 + a9/Tr**3) / Vr**4
            term4 = (a10 + a11/Tr**2 + a12/Tr**3) / Vr**5
            term5 = a13 / (Tr**3 * Vr**2) * (a14 + a15 / Vr**2) * np.exp(-a15 / Vr**2)
            return Pr * Vr / Tr - (1 + term1 + term2 + term3 + term4 + term5)
        
        def solve_Vr(Pr, Tr, Vr_initial_guess=1.0):
            Vr_solution = newton(equation_Vr, Vr_initial_guess, args=(Pr, Tr))
            return Vr_solution
        
        def calculate_Z(Pr, Tr, Vr):
            Z = Pr * Vr / Tr
            return Z
        
        def calculate_phi_co2(Z, Tr, Vr):
            a1 = 8.99288497e-2
            a2 = -4.94783127e-1
            a3 = 4.77922245e-2
            a4 = 1.03808883e-2
            a5 = -2.82516861e-2
            a6 = 9.49887563e-2
            a7 = 5.20600880e-4
            a8 = -2.93540971e-4
            a9 = -1.77265112e-3
            a10 = -2.51101973e-5
            a11 = 8.93353441e-5
            a12 = 7.88998563e-5
            a13 = -1.66727022e-2
            a14 = 1.39800000
            a15 = 2.96000000e-2
            term1 = (a1 + a2/Tr**2 + a3/Tr**3) / Vr
            term2 = (a4 + a5/Tr**2 + a6/Tr**3) / (2 * Vr**2)
            term3 = (a7 + a8/Tr**2 + a9/Tr**3) / (4 * Vr**4)
            term4 = (a10 + a11/Tr**2 + a12/Tr**3) / (5 * Vr**5)
            term5 = a13 / (2 * Tr**3 * a15) * (a14 + 1 - (a14 + 1 + a15 / Vr**2) * np.exp(-a15 / Vr**2))
            ln_phi_co2 = Z - 1 - np.log(Z) + term1 + term2 + term3 + term4 + term5
            return np.exp(ln_phi_co2)
        
        def calculate_mu_l0_co2_RT(P, T):
            c1 = 28.9447706
            c2 = -0.0354581768
            c3 = -4770.67077
            c4 = 1.02782768e-5
            c5 = 33.8126098
            c6 = 9.04037140e-3
            c7 = -1.14934031e-3
            c8 = -0.307405726
            c9 = -0.0907301486
            c10 = 9.32713393e-4
            mu_l0_co2_RT = c1 + c2 * T + c3 / T + c4 * T**2 + c5 / (630 - T) + c6 * P + c7 * P * np.log(T) + c8 * P / T \
                            + c9 * P / (630 - T) + c10 * P**2 / (630 - T)**2
            return mu_l0_co2_RT
        
        def calculate_lambda_co2_na(P, T):
            c1 = -0.411370585
            c2 = 6.07632013e-4
            c3 = 97.5347708
            c8 = -0.0237622469
            c9 = 0.0170656236
            c11 = 1.41335834e-5
            lambda_co2_Na = c1 + c2 * T + c3 / T + c8 * P / T + c9 * P / (630 - T) + c11 * T * np.log(P)
            return lambda_co2_Na
        
        def calculate_xi_co2_na_cl(P, T):
            c1 = 3.36389723e-4
            c2 = -1.98298980e-5
            c8 = 2.12220830e-3
            c9 = -5.24873303e-3
            xi_co2_na_cl = c1 + c2 * T + c8 * P / T + c9 * P / (630 - T)
            return xi_co2_na_cl
        
        def calculate_m_co2(P, T, m_ca, m_cl, m_na, m_K, m_mg, m_so4):
            lambda_co2_na = calculate_lambda_co2_na(P, T)
            xi_co2_na_cl = calculate_xi_co2_na_cl(P, T)
            mu_l0_co2_RT = calculate_mu_l0_co2_RT(P, T)
            P_H2O = calculate_P_H2O(T)
            y_co2 = calculate_y_CO2(P, P_H2O)
            Pr = calculate_Pr(P)
            Tr = calculate_Tr(T)
            Vr = solve_Vr(Pr, Tr)
            Z = calculate_Z(Pr, Tr, Vr)
            phi_co2 = calculate_phi_co2(Z, Tr, Vr)
            ln_m_co2 = (np.log(y_co2 * phi_co2 * P)
                        - mu_l0_co2_RT
                        - 2 * lambda_co2_na * (m_na + m_k + 2 * m_ca + 2 * m_mg)
                        - xi_co2_na_cl * m_cl * (m_na + m_k + m_ca + m_mg)
                        + 0.07 * m_so4)
            
            return np.exp(ln_m_co2)
            
               
        P =row['Pressure']*10 #bar
        T = row['Temperature'] #kelvin
        m_ca = row['CaCl2 Concentration']
        m_cl = 2 * row['CaCl2 Concentration'] + row['NaCl Concentration']+ row['KCl Concentration'] + 2 * row['MgCl2 Concentration']
        m_na =  row['NaCl Concentration']
        m_k = row['KCl Concentration']
        m_mg = row['MgCl2 Concentration']
       
        # Calculate CO2 solubility
        try:
            co2_solubility = calculate_m_co2(P, T, m_ca , m_cl, m_na, m_k, m_mg, 0)
        except RuntimeError:
            co2_solubility=0
            pass
            

        return co2_solubility

    # Calculate CO2 solubility for each row
    df["Sol_Duansun"] = df.apply(get_co2_solubility_Duansun, axis=1)
    
    print("CO2 solubility calculation completed.")
    return df  


# In[ ]:


# ERROR Calculation
def calculate_aad(actual, predicted):
    """Calculate Average Absolute Deviation between actual and predicted values."""
    return np.mean(np.abs(actual - predicted)/actual*100)


def calculate_aad_for_data(df, actual_col='CO2 Solubility'):
    # Get the list of unique paper titles
  
    # Get the list of columns that contain solubility predictions
    solubility_columns = [col for col in df.columns if 'Sol_' in col ]

    # Initialize a dictionary to store AAD values
    aad_data_results = df.copy()
    
    
    for col in solubility_columns:
        #print(f"  Using method: {col}")
            aad = np.abs((df[actual_col] - df[col])/df[actual_col]*100)
            # Store AAD as percentage
            aad_data_results[col] = aad
             
    print("AAD calculation completed.")
    return aad_data_results

def calculate_aad_for_papers(df, actual_col='CO2 Solubility'):
    # Get the list of unique paper titles
    papers = df['Paper Title'].unique()
    
    # Get the list of columns that contain solubility predictions
    solubility_columns = [col for col in df.columns if 'Sol_' in col ]
    # Initialize a dictionary to store AAD values
    aad_results = {paper: {} for paper in papers}
    
    for paper in papers:
        #print(f"Calculating AAD for paper: {paper}")
        
        # Filter the dataframe for the current paper
        paper_df = df[df['Paper Title'] == paper]
        
        for col in solubility_columns:
            #print(f"  Using method: {col}")
            
            # Calculate AAD for the current column
            aad = calculate_aad(paper_df[actual_col], paper_df[col])
            
            # Store AAD as percentage
           # aad_results[paper][col] = f"{aad * 100:.2f}%"  # Format as percentage string
            aad_results[paper][col] = aad 
    print("AAD calculation completed.")
    return aad_results


   


# In[ ]:


# Main code
import pandas as pd
import os

current_directory = os.getcwd()
parent_directory = os.path.split(current_directory)[0]
file_path=os.path.join(parent_directory, 'datasets\preprocessedDataset.csv')

clean_df = read_and_rename_columns(file_path)

# Export clean_df to Excel
clean_df.to_excel('clean_dataset.xlsx', index=False)
print("The file 'clean_dataset.xlsx' has been written.")


# In[ ]:


df_with_solubility_Duansun= calculate_co2_solubility_Duansun(clean_df)


# In[ ]:


# Save the calculation results to an Excel file

df_with_solubility_Duansun.to_excel('CO2_solubility_Duansun_calculations.xlsx', index=False)
print("The file 'CO2_solubility_Duansun_calculations.xlsx' has been written.")

df_with_solubility_Duansun=df_with_solubility_Duansun[df_with_solubility_Duansun["Sol_Duansun"]!=0]


aad_data = calculate_aad_for_data(df_with_solubility_Duansun)
# Save the calculation results to an Excel file with AAD as percentage
aad_data.to_excel('aad_data_Duansun_results.xlsx')
print("The file 'aad_data_Duansun_results.xlsx' has been written with AAD as percentage.")



aad_results = calculate_aad_for_papers(df_with_solubility_Duansun)
aad_df = pd.DataFrame(aad_results).T
# Save the calculation results to an Excel file with AAD as percentage
aad_df.to_excel('aad_Duansun_results.xlsx')
print("The file 'aad_Duansun_results.xlsx' has been written with AAD as percentage.")




# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Box plot

solubility_columns = [col for col in aad_data.columns if 'Sol_' in col ]
#solubility_columns ="Sol_Duansun"
papers = aad_data['Paper Title'].unique()

# Initialize a dictionary to store AAD values
Boxplotdata = {paper: {} for paper in papers}

for paper in papers:
    Boxplotdata[paper]["mean-o"]=np.mean(aad_data[aad_data['Paper Title'] == paper][solubility_columns])
    Boxplotdata[paper]["max"]=np.max(aad_data[aad_data['Paper Title'] == paper][solubility_columns])
    Boxplotdata[paper]["min"]=np.min(aad_data[aad_data['Paper Title'] == paper][solubility_columns])
    Boxplotdata[paper]["mean-c"]=np.mean(aad_data[aad_data['Paper Title'] == paper][solubility_columns])


Boxplotdata_df = pd.DataFrame(Boxplotdata).T
# Save the calculation results to an Excel file with AAD as percentage
Boxplotdata_df.to_excel('Boxplotdata.xlsx')
print("The file 'Boxplotdata.xlsx' has been written with AAD as percentage.")
    

