#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data preparation 
import pandas as pd
import numpy as np
from reaktoro import *

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
def read_and_rename_columns(file_path1,file_path2,file_path3):
    # Define the actual column names as they appear in the Excel file
    columns_to_read = ['Paper Title', 'Temperature', 'Temperature Unit', 'Pressure', 'Pressure Unit','CaCl2 Concentration', 'NaCl Concentration', 'KCl Concentration','MgCl2 Concentration', 'Concentration Unit', 'CO2 Solubility', 'Solubility Unit']
    columns_to_read1=['HCl(%)','Molality' ,'HCl','H2O','MgCl2','CaCl2','CO2']



    salt_columns = ['CaCl2 Concentration', 'NaCl Concentration', 'KCl Concentration', 'MgCl2 Concentration']
    
    # Read the Excel file
    df = pd.read_csv(file_path1 ,usecols=columns_to_read)
    df1=pd.read_csv(file_path2,usecols=columns_to_read1 )
    df2=pd.read_csv(file_path3,usecols=columns_to_read1 )

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
    
    return df,df1,df2


# In[ ]:


#  Co2 solubility Calculation
def calculate_co2_solubility(df):
    df=df.copy()
    databases = {
        "ThermoFun": ("aq17", "H2O@", "CO2@"),
        "Phreeqc": ("phreeqc.dat", "H2O", "CO2"),
        "Supcrt": ("supcrtbl", "H2O(aq)", "CO2(aq)")
    }

    # Define the activity models with corresponding string names
    solution_activity_models = {
        "HKF": ActivityModelHKF(),
        "Pitzer": ActivityModelPitzer(),
        "Davies": ActivityModelDavies(),
        "DebyeHuckel": ActivityModelDebyeHuckel(),
        "DebyeHuckelKielland": ActivityModelDebyeHuckelKielland(),
        "DebyeHuckelLimitingLaw": ActivityModelDebyeHuckelLimitingLaw()
    }


    #"Rumpf": ActivityModelRumpf("CO2"),
    gaseous_activity_models = {
        "SpycherPruessEnnis": ActivityModelSpycherPruessEnnis(),
        
    }
    #"SpycherReed": ActivityModelSpycherReed()
    # Iterate over all combinations of databases and activity models
    for db_name, (db_file, h2o_label, co2_label) in databases.items():
        print(f"Using database: {db_name}")
        
        if db_name == "ThermoFun":
            db = ThermoFunDatabase(db_file)
            S=" H+ OH- H2O@ Mg+2 Cl- Na+ K+ Ca+2 CO2@ "
        elif db_name == "Phreeqc":
            db = PhreeqcDatabase(db_file)
            S=" H+ OH- H2O Mg+2 Cl- Na+ K+ Ca+2 CO2 "
        elif db_name == "Supcrt":
            db = SupcrtDatabase(db_file)
            S=" H+ OH- H2O(aq) Mg+2 Cl- Na+ K+ Ca+2 CO2(aq) "
            
        for sol_name, sol_model in solution_activity_models.items():
                for gas_name, gas_model in gaseous_activity_models.items():
                    column_name = f'{db_name}_Sol_{sol_name}'
                    print(f"  Calculating solubility using method: {column_name}")


                    
                    
                    def get_co2_solubility(row):
                     
                        solution = AqueousPhase(S)
                        solution.setActivityModel(sol_model)
                    
                        # Conditionally set the gaseous phase based on the database
                        if db_name == "Phreeqc" or db_name == "Supcrt":
                            gases = GaseousPhase("CO2(g) ")
                        else:
                            gases = GaseousPhase("CO2 ")
                            
                        gases.setActivityModel(gas_model)

                   
                        if row['Concentration Unit']=="mol/kg":


                            system = ChemicalSystem(db, solution, gases)

                              # Set the concentrations of salts
                            state = ChemicalState(system)

                            
                            # Set the concentrations of salts
                            state.set(co2_label, 1, "kg")
                            # Set the concentrations of salts
                            if row['CaCl2 Concentration'] != 0:
                                state.set("Ca+2", row['CaCl2 Concentration'], "mol")
                                state.add("Cl-", 2 * row['CaCl2 Concentration'], "mol")
                            if row['NaCl Concentration'] != 0:
                                state.set("Na+", row['NaCl Concentration'], "mol")
                                state.add("Cl-", row['NaCl Concentration'], "mol")
                            if row['KCl Concentration'] != 0:
                                state.set("K+", row['KCl Concentration'], "mol")
                                state.add("Cl-", row['KCl Concentration'], "mol")
                            if row['MgCl2 Concentration'] != 0:
                                state.set("Mg+2", row['MgCl2 Concentration'], "mol")
                                state.add("Cl-", 2 * row['MgCl2 Concentration'], "mol")
                           
                            
                            # Set the temperature and pressure
                            state.temperature(row['Temperature'], "kelvin")
                            state.pressure(row['Pressure'], "MPa")
                            state.set(h2o_label, 1, "kg")
                            
                            solver = EquilibriumSolver(system)
                            solver.solve(state)

                            # Calculate CO2 solubility
                            props = ChemicalProps(state)                       
                            co2_solubility = props.speciesAmount(co2_label) / props.speciesMass(h2o_label)
                        else:

                            molarity=0
                            molality= row['CaCl2 Concentration']
                            try:
                                while  np.abs(molarity-row['CaCl2 Concentration']) >=0.1:                              
                                    system = ChemicalSystem(db, solution, gases)
                                    
                                      # Set the concentrations of salts
                                    state = ChemicalState(system)
                                         # Set the temperature and pressure
                                    state.temperature(row['Temperature'], "kelvin")
                                    state.pressure(row['Pressure'], "MPa")
                                
                                    if row['CaCl2 Concentration'] != 0:
                                        state.set("Ca+2", molality, "mol")
                                        state.add("Cl-", 2 * molality, "mol")
                         
                                    state.set(h2o_label, 1, "kg")                            
                                    state.set(co2_label, 1, "kg")
                                    
                                    # Define equilibrium solver and equilibrate given initial state
                                    solver = EquilibriumSolver(system)
                                    solver.solve(state)
                                    state.scalePhaseVolume("AqueousPhase", 1000, "cm3")
                                    
                                    aprops = AqueousProps(state)
                                    props = ChemicalProps(state)
                                    molarity=props.elementAmount("Ca")
                                    molality=molality+0.1
                                    Co2Amount=props.speciesAmount(co2_label)
                                    #print(molarity)
                            except RuntimeError:
                                Co2amount=0
                                pass

                            
                            # Calculate CO2 solubility
                            co2_solubility = Co2Amount / props.speciesMass(h2o_label)
                         
                        return co2_solubility

                    # Calculate CO2 solubility for each row
                    df[column_name] = df.apply(get_co2_solubility, axis=1)
  
    print("CO2 solubility calculation completed.")
    return df 


# In[ ]:


# Effect of HCl on CO2 Solubility 
def calculate_co2_solubility_HCl(df_HCl):
    Pressure=[30,40,50,60]
    Temperature=[300,350,400,450]
    df_HCl=df_HCl.copy()
    for P in Pressure:
        for T in Temperature:
            for e in range(0,2):


                db = ThermoFunDatabase("aq17")
                
                S=" H+ OH- H2O@ Mg+2 Cl- Na+ K+ Ca+2 CO2@ HCl@"

                h2o_label= "H2O@"
                co2_label="CO2@"
                solution = AqueousPhase(S)
                solution.setActivityModel(ActivityModelHKF())
                
                # Conditionally set the gaseous phase based on the database
               
                gases = GaseousPhase("CO2")
                    
                gases.setActivityModel(ActivityModelSpycherPruessEnnis())
          
                column_name =["Not Considering HCl","Considering HCl"]
            
            
                def get_co2_solubility_HCL(row):
                   
                    system = ChemicalSystem(db, solution, gases)
        
                      # Set the concentrations of salts
                    state = ChemicalState(system)
            
                
                    # Set the concentrations of salts
                    state.set(co2_label,  1, "kg")
                    # Set the concentrations of salts
                    
                    state.set("Ca+2", row['CaCl2'], "mol")
                    
                    state.set("Mg+2", row['MgCl2'], "mol")
                    
                    state.set("Cl-",   2 * row['CaCl2']  + 2 * row['MgCl2'], "mol")

                    if e==0:
                        state.set("HCl@",0, "mol")
                        
                    else:
                        state.set("HCl@", row['HCl'], "mol")
                   
                    state.set(h2o_label, row['H2O'], "mol")
                    # Set the temperature and pressure
                    state.temperature(T, "kelvin")
                    state.pressure(P, "MPa")
                    solver = EquilibriumSolver(system)
                    solver.solve(state)
                    # Calculate CO2 solubility
                    props = ChemicalProps(state)                       
                    co2_solubility = props.speciesAmount(co2_label) / props.speciesMass(h2o_label)
                    return co2_solubility
                        
            
            # Calculate CO2 solubility for each row
    
                df_HCl[f"{column_name[e]} @ {T} K-{P} MPa"] = df_HCl.apply(get_co2_solubility_HCL, axis=1)
      
    print("CO2 solubility calculation completed.")
    return df_HCl


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
file_path1=os.path.join(parent_directory, 'datasets\preprocessedDataset.csv')
file_path2=os.path.join(parent_directory, 'stoichiometry\calciteDissolution.csv')
file_path3=os.path.join(parent_directory, 'stoichiometry\dolomiteDissolution.csv')

clean_df,df_HCl_Carbonate,df_HCl_Dolomite = read_and_rename_columns(file_path1,file_path2,file_path3)

# Export clean_df to Excel
clean_df.to_excel('clean_dataset.xlsx', index=False)
print("The file 'clean_dataset.xlsx' has been written.")


# In[ ]:


df_with_solubility = calculate_co2_solubility(clean_df)
df_HCl_with_solubility_Carbonate = calculate_co2_solubility_HCl(df_HCl_Carbonate)
df_HCl_with_solubility_Dolomite = calculate_co2_solubility_HCl(df_HCl_Dolomite) 


# In[ ]:


# Save the calculation results to an Excel file
df_with_solubility.to_excel('CO2_solubility_calculations.xlsx', index=False)
print("The file 'CO2_solubility_calculations.xlsx' has been written.")



df_HCl_with_solubility_Carbonate.to_excel('CO2_solubility_HCL_Carbonate_calculations.xlsx', index=False)
print("The file 'CO2_solubility_HCL_Carbonate_calculations.xlsx' has been written.")

df_HCl_with_solubility_Dolomite.to_excel('CO2_solubility_HCL_Dolomite_calculations.xlsx', index=False)
print("The file 'CO2_solubility_HCL_Dolomite_calculations.xlsx' has been written.")


aad_data = calculate_aad_for_data(df_with_solubility)
# Save the calculation results to an Excel file with AAD as percentage
aad_data.to_excel('aad_data_results.xlsx')
print("The file 'aad_data_results.xlsx' has been written with AAD as percentage.")



aad_results = calculate_aad_for_papers(df_with_solubility)
aad_df = pd.DataFrame(aad_results).T
# Save the calculation results to an Excel file with AAD as percentage
aad_df.to_excel('aad_results.xlsx')
print("The file 'aad_results.xlsx' has been written with AAD as percentage.")





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

#Box plot

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
    

#T & p Range plot for methods
# Initialize a dictionary to store AAD values


Boxplotdata_T1 = {method: {} for method in solubility_columns}

for method in solubility_columns:
    Boxplotdata_T1[method]["297-350"]=np.mean(aad_data[(aad_data['Temperature'] <=350)][method])
    Boxplotdata_T1[method]["350-400"]=np.mean(aad_data[(aad_data['Temperature'] <=400) & (aad_data['Temperature'] >350)][method])
    Boxplotdata_T1[method]["400-453.15"]=np.mean(aad_data[(aad_data['Temperature'] >400)][method])


Boxplotdata_T1_df = pd.DataFrame(Boxplotdata_T1).T
# Save the calculation results to an Excel file with AAD as percentage
Boxplotdata_T1_df.to_excel('Boxplotdata_T1.xlsx')
print("The file 'Boxplotdata_T1.xlsx' has been written with AAD as percentage.")
    
Boxplotdata_P1 = {method: {} for method in solubility_columns}

for method in solubility_columns:
    Boxplotdata_P1[method]["0-30"]=np.mean(aad_data[(aad_data['Pressure'] <=30) ][method])
    Boxplotdata_P1[method]["30-60"]=np.mean(aad_data[ (aad_data['Pressure'] <=60) & (aad_data['Pressure'] >30)][method])
    Boxplotdata_P1[method]["60-90"]=np.mean(aad_data[ (aad_data['Pressure'] >60) ][method])


Boxplotdata_P1_df = pd.DataFrame(Boxplotdata_P1).T
# Save the calculation results to an Excel file with AAD as percentage
Boxplotdata_P1_df.to_excel('Boxplotdata_P1.xlsx')
print("The file 'Boxplotdata_P1.xlsx' has been written with AAD as percentage.")


# Heat map
plt.figure(figsize=(16,8))
hm=sns.heatmap(aad_df, annot=True,cmap="RdYlGn_r",fmt=".01f", linewidth=.5)


# Improve layout  
plt.tight_layout()


# Save the figure
plt.savefig('Heatmap.png', dpi=300, bbox_inches='tight')
print("The heatmap has been plotted.")



# In[ ]:




