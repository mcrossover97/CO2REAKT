#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
from scipy.optimize import newton
from scipy.interpolate import interp1d
import time

# Data from the table (MPa and corresponding temperatures in Kelvin)
pressure_data = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 
                              3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 
                              4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 
                          6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 
                          7.0, 7.1, 7.2, 7.3, 7.377])  # Pressure in MPa

temperature_data = np.array([-19.50, -17.90, -16.36, -14.86, -13.42, -12.01, -10.65, -9.32, 
                                -8.03, -6.78, -5.55, -4.36, -3.19, -2.05, -0.93, 0.16, 1.23, 
                                2.28, 3.30, 4.31, 5.30, 6.27, 7.22, 8.16, 9.08, 9.98, 10.87, 
                                11.74, 12.60, 13.45, 14.28, 15.12, 15.91, 16.71, 17.50, 18.27, 19.03, 19.78, 20.53, 21.26, 
                              21.98, 22.69, 23.39, 24.08, 24.77, 25.44, 26.11, 26.77, 27.41, 28.05, 
                              28.68, 29.30, 29.92, 30.52, 30.98]) + 273.15  # Convert to Kelvin

saturation_pressure_interp = interp1d(temperature_data, pressure_data, kind="linear", fill_value="extrapolate")

def saturation_pressure_CO2(T):
    if T < temperature_data.min() or T > temperature_data.max():
        raise ValueError(f"Temperature {T} K is out of range ({temperature_data.min()} K to {temperature_data.max()} K).")
    return float(saturation_pressure_interp(T)) * 10

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

def get_P1(T):
    """Determine P1 based on temperature T."""
    if T < 305:
        P1 = saturation_pressure_CO2(T)  # Replace with the actual function or value for CO2 saturation pressure
    elif 305 <= T < 405:
        P1 = 75 + (T - 305) * 1.25
    else:  # T >= 405
        P1 = 200
    return P1

def calculate_new_phi_co2(P, T):
    P1 = get_P1(T)
    # Condition checks
    if 273 < T < 573 and P < P1:
        c1  = 1.0
        c2  = 4.7586835E-3
        c3  = -3.3569963E-6
        c4  = 0.0
        c5  = -1.3179396
        c6  = -3.8389101E-6
        c7  = 0.0
        c8  = 2.2815104E-3
        c9  = 0.0
        c10 = 0.0
        c11 = 0.0
        c12 = 0
        c13 = 0
        c14 = 0
        c15 = 0
    elif 273 < T < 340 and P1 < P < 1000:
        c1  = -7.1734882E-1
        c2  = 1.5985379E-4
        c3  = -4.9286471E-7
        c4  = 0.0
        c5  = 0.0
        c6  = -2.7855285E-7
        c7  = 1.1877015E-9
        c8  = 0.0
        c9  = 0.0
        c10 = 0.0
        c11 = 0.0
        c12 = -96.539512
        c13 = 4.4774938E-1
        c14 = 101.81078
        c15 = 5.3783879E-6
    elif 273 < T < 340 and P > 1000:
        c1  = -6.5129019E-2
        c2  = -2.1429977E-4
        c3  = -1.1444930E-6
        c4  = 0.0
        c5  = 0.0
        c6  = -1.1558081E-7
        c7  = 1.1952370E-9
        c8  = 0.0
        c9  = 0.0
        c10 = 0.0
        c11 = 0.0
        c12 = -221.34306
        c13 = 0.0
        c14 = 71.820393
        c15 = 6.6089246E-6
    elif 340 <= T < 435 and P1 < P <= 1000:
        c1  = 5.0383896
        c2  = -4.4257744E-3
        c3  = 0.0
        c4  = 1.9572733
        c5  = 0.0
        c6  = 2.4223436E-6
        c7  = 0.0
        c8  = -9.3796135E-4
        c9  = -1.5026030
        c10 = 3.0272240E-3
        c11 = -31.377342
        c12 = -12.847063
        c13 = 0.0
        c14 = 0.0
        c15 = -1.5056648E-5
    elif 340 <= T < 435 and P > 1000:
        c1  = -16.063152
        c2  = -2.7057990E-3
        c3  = 0.0
        c4  = 1.4119239E-1
        c5  = 0.0
        c6  = 8.1132965E-7
        c7  = 0.0
        c8  = -1.1453082E-4
        c9  = 2.3895671
        c10 = 5.0527457E-4
        c11 = -17.763460
        c12 = 985.92232
        c13 = 0.0
        c14 = 0.0
        c15 = -5.4965256E-7
    elif T > 435 and P > P1:
        c1 = -1.5693490E-1
        c2 = 4.4621407E-4
        c3 = -9.1080591E-7
        c4 = 0
        c5 = 0
        c6 = 1.0647399E-7
        c7 = 2.4273357E-10
        c8 = 0
        c9 = 3.5874255E-1
        c10 = 6.3319710E-5
        c11 = -249.89661
        c12 = 0
        c13 = 0
        c14 = 888.76800
        c15 = -6.6348003E-7
    else:
        print("No conditions met.")    
    
    phi_co2 = c1 + (c2+c3*T+c4/T+c5/(T-150))*P + (c6+c7*T+c8/T)*P**2 + (c9+c10*T+c11/T)*np.log(P) + (c12 + c13*T)/P + c14/T + c15*T**2
    return phi_co2

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
    phi_new_co2 = calculate_new_phi_co2(P, T)
    ln_m_co2 = (np.log(y_co2 * phi_new_co2 * P)
                - mu_l0_co2_RT
                - 2 * lambda_co2_na * (m_na + m_k + 2 * m_ca + 2 * m_mg)
                - xi_co2_na_cl * m_cl * (m_na + m_k + m_ca + m_mg)
                + 0.07 * m_so4)
    
    return np.exp(ln_m_co2)

# Example usage
P = 2000 #bar
T = 393.15 #kelvin
m_ca = 0
m_cl = 0
m_na = 0
m_k = 0
m_mg = 0
m_so4 = 0

start_time = time.time()

for _ in range(100000):
    m_co2 = calculate_m_co2(P, T,  m_ca, m_cl, m_na, m_k, m_mg, m_so4)
    print("m_co2 =", m_co2)

    end_time = time.time()
    end_time

end_time - start_time


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# In[20]:


# Step 1: Read the data from an Excel file or CSV file
# Update the file path as needed
file_path = 'CO2_solubility_Duansun_calculations.csv'  # Change this to your file path
df = pd.read_csv(file_path)  # or pd.read_csv(file_path) for CSV files


# In[25]:


# Step 2: Extract the MSE and MAE values from columns O and N
# Assuming the columns are labeled 'O' and 'N', if they have different names, replace accordingly
mse = df['MSE']  # Column O
mae = df['MAE']  # Column N


# In[26]:


# Step 3: Plot the values against their indices
plt.figure(figsize=(14, 6))


# In[35]:


import matplotlib.pyplot as plt
import pandas as pd

# Create a figure with a larger size and higher DPI
plt.figure(figsize=(12, 8), dpi=100)

# Plot MSE with just dots, using a larger dot size and a different color with decreased opacity
plt.plot(mse.index, mse, marker='o', linestyle='', markersize=10, label='MSE', color='royalblue', alpha=0.5)

# Enhancing the title and labels with a larger font size
plt.xlabel('Index', fontsize=16)
plt.ylabel('MSE Value', fontsize=16)

# Adding grid lines for better readability
plt.grid(color='gray', linestyle='--', linewidth=0.7)

# Adding a legend with a larger font size
plt.legend(fontsize=14)

# Setting x and y limits for better visualization (optional)
plt.xlim(mse.index.min() - 0.5, mse.index.max() + 0.5)
plt.ylim(mse.min()-0.1, mse.max() + 1)

# Adding a background color to the plot
plt.gca().set_facecolor('whitesmoke')

# Show the plot
plt.tight_layout()  # Adjust layout for better fit
plt.show()


# In[48]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'mse' is a pandas Series
# Calculate Z-scores
mean_mse = mse.mean()
std_mse = mse.std()
threshold = 3  # Common threshold for identifying outliers
z_scores = (mse - mean_mse) / std_mse

# Identify outliers
outliers_z = mse[np.abs(z_scores) > threshold]

# Print outliers with their index and values
print("Outliers:")
for index, value in outliers_z.items():
    print(f"Index: {index}, Value: {value}")

# Plot MSE with outliers highlighted
plt.figure(figsize=(12, 8), dpi=100)
plt.plot(mse.index, mse, marker='o', linestyle='', markersize=10, label='MSE', color='royalblue', alpha=0.5)
plt.plot(outliers_z.index, outliers_z, marker='o', linestyle='', markersize=10, label='Outliers', color='red')

plt.xlabel('Index', fontsize=20)  # Increased font size for x-axis
plt.ylabel('MSE Value', fontsize=20)  # Increased font size for y-axis
plt.grid(color='gray', linestyle='--', linewidth=0.7)
plt.legend(fontsize=16)
plt.xlim(mse.index.min() - 0.5, mse.index.max() + 0.5)
plt.ylim(mse.min()-0.1, mse.max() + 1)
plt.gca().set_facecolor('whitesmoke')

# Increase the size of tick labels
plt.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label font size

plt.tight_layout()
plt.show()


# In[ ]:


# After removing the outliers MSE = 0.002474
#Before removing the outliers MSE = 0.005884


# In[36]:


import matplotlib.pyplot as plt
import pandas as pd

# Create a figure with a larger size and higher DPI
plt.figure(figsize=(12, 8), dpi=100)

# Plot MSE with just dots, using a larger dot size and a different color
plt.plot(mae.index, mae, marker='o', linestyle='', markersize=10, label='MAE', color='orange',alpha=0.5)

# Enhancing the title and labels with a larger font size
plt.xlabel('Index', fontsize=16)
plt.ylabel('MAE Value', fontsize=16)

# Adding grid lines for better readability
plt.grid(color='gray', linestyle='--', linewidth=0.7)

# Adding a legend with a larger font size
plt.legend(fontsize=14)

# Setting x and y limits for better visualization (optional)
plt.xlim(mae.index.min() - 0.5, mae.index.max() + 0.5)
plt.ylim(mae.min()-0.1, mae.max() + 1)

# Adding a background color to the plot
plt.gca().set_facecolor('whitesmoke')

# Show the plot
plt.tight_layout()  # Adjust layout for better fit
plt.show()


# In[51]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'mse' is a pandas Series
# Calculate Z-scores
mean_mae = mae.mean()
std_mae = mae.std()
threshold = 3  # Common threshold for identifying outliers
z_scores = (mae - mean_mae) / std_mae

# Identify outliers
outliers_z = mae[np.abs(z_scores) > threshold]

# Print outliers with their index and values
print("Outliers:")
for index, value in outliers_z.items():
    print(f"Index: {index}, Value: {value}")

# Plot MSE with outliers highlighted
plt.figure(figsize=(12, 8), dpi=100)
plt.plot(mae.index, mae, marker='o', linestyle='', markersize=10, label='MAE', color='orange', alpha=0.5)
plt.plot(outliers_z.index, outliers_z, marker='o', linestyle='', markersize=10, label='Outliers', color='red')

plt.xlabel('Index', fontsize=20)
plt.ylabel('MAE Value', fontsize=20)
plt.grid(color='gray', linestyle='--', linewidth=0.7)
plt.legend(fontsize=16)
plt.xlim(mae.index.min() - 0.5, mae.index.max() + 0.5)
plt.ylim(mae.min()-0.1, mae.max() + 1)

# Increase the size of tick labels
plt.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label font size

plt.gca().set_facecolor('whitesmoke')
plt.tight_layout()
plt.show()


# In[ ]:


# After removing the outliers MAE = 0.025605
#Before removing the outliers MAE = 0.03443


# In[ ]:





# In[ ]:





# In[ ]:




