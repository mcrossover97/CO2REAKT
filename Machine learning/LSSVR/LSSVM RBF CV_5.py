#!/usr/bin/env python
# coding: utf-8

# In[17]:


# Pandas is used for data manipulation
import pandas as pd# Read in data and display first 5 rows
df = pd.read_csv('../../datasets/Normilized_dataset.csv')

# Use numpy to convert to arrays
import numpy as np
import matplotlib. pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 8, 8

# Data preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 
from sklearn.metrics import r2_score
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error , mean_squared_error
from sklearn.model_selection import GridSearchCV
#importing Progressbar
from tqdm import tqdm


# In[18]:


#importing LSSVM libraries
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error


# In[3]:


print('The shape of our features is:', df.shape)


# In[4]:


# Descriptive statistics for each column
df.describe()


# In[5]:


df.info()


# In[6]:


# view normalized data 
print(df)


# In[7]:


# Labels are the values we want to predict
label_data = np.array(df['CO2 Solubility (mol/kg)'])


# In[8]:


label_data.shape


# In[9]:


# Remove the labels from the features
# axis 1 refers to the columns
Train_data= df.drop('CO2 Solubility (mol/kg)', axis = 1)


# In[10]:


Train_data.shape


# In[11]:


# Saving feature names for later use
feature_list = list(Train_data.columns)


# In[12]:


feature_list


# In[13]:


# 20% for validation
x_train, x_test, y_train, y_test = train_test_split(Train_data, label_data, test_size=0.2, random_state=12) # 0.20 x 1 = 0.20




# In[14]:


# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
   'epsilon': [0.01, 0.1, 0.2, 0.5, 1.0],
    'kernel': ['rbf'],
   'gamma': [0.01,0.1, 1.0, 10,100]
    
}


# In[15]:


# Create an LSSVM model
lssvm_model = SVR()


# In[16]:


# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(lssvm_model, param_grid, cv=5, scoring='neg_mean_squared_error')



# In[20]:


grid_search.fit(x_train, y_train)


# In[21]:


# Display the fitting scores for each combination of hyperparameters
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']


# In[22]:


# Get the best hyperparameters
best_params = grid_search.best_params_
print(f'\nBest Hyperparameters: {best_params}')


# In[23]:


# Print the total number of fittings
total_fittings = len(means)
print(f'Total number of fittings: {total_fittings}')


# In[24]:


# Make predictions on the test set using the best model
best_lssvm_model = grid_search.best_estimator_


# In[19]:


from joblib import dump
dump(best_lssvm_model, 'best_lssvm_model.joblib')


# In[20]:


from joblib import load

# Load the model
best_lssvm_model = load('best_lssvm_model.joblib')


# In[21]:


# metrics and prediction of test data
testpreds = best_lssvm_model.predict(x_test)


# In[22]:


print('test mae = ',mean_absolute_error(y_test, testpreds)) # train
print('test mse = ',mean_squared_error(y_test, testpreds)) # train


# In[23]:


acc_test = r2_score(y_test, testpreds)
acc_test


# In[24]:


# metrics and prediction of train data
trainpreds = best_lssvm_model.predict(x_train)


# In[25]:


print('train mae = ',mean_absolute_error(y_train, trainpreds)) # train
print('train mse = ',mean_squared_error(y_train, trainpreds)) # train


# In[26]:


acc_train = r2_score(y_train, trainpreds)
acc_train


# In[27]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot for Training Results
sns.regplot(x=y_train, y=trainpreds, scatter_kws={'alpha': 0.5, 'color': 'purple'}, line_kws={'color': 'purple'}, ci=0, ax=axes[0, 0], label="Train Regression Line")
sns.scatterplot(x=y_train, y=trainpreds, color='blue', alpha=0.5, ax=axes[0, 0], label="Train Original Data")
axes[0, 0].set_xlabel("Actual CO2 Solubility (mol/kg)", fontsize=10)
axes[0, 0].set_ylabel("Predicted CO2 Solubility (mol/kg)", fontsize=10)
axes[0, 0].set_title("Training Results")
axes[0, 0].legend(loc='upper left')
# Add R-squared to the bottom right
r_squared_train = r2_score(y_train, trainpreds)
axes[0, 0].text(0.95, 0.05, f'R-squared = {r_squared_train:.3f}', ha='right', va='bottom', transform=axes[0, 0].transAxes)

# Plot for Testing Results
sns.regplot(x=y_test, y=testpreds, scatter_kws={'alpha': 0.5, 'color': 'green'}, line_kws={'color': 'green'}, ci=0, ax=axes[0, 1], label="Testing Regression Line")
sns.scatterplot(x=y_test, y=testpreds, color='orange', alpha=0.5, ax=axes[0, 1], label="Validation Original Data")
axes[0, 1].set_xlabel("Actual CO2 Solubility (mol/kg)", fontsize=10)
axes[0, 1].set_ylabel("Predicted CO2 Solubility (mol/kg)", fontsize=10)
axes[0, 1].set_title("Testing Results")
axes[0, 1].legend(loc='upper left')
# Add R-squared to the bottom right
r_squared_val = r2_score(y_test, testpreds)
axes[0, 1].text(0.95, 0.05, f'R-squared = {r_squared_val:.3f}', ha='right', va='bottom', transform=axes[0, 1].transAxes)


# Set the DPI (dots per inch)
plt.figure(dpi=300)

# tight layout
fig.tight_layout()
# show the plot
plt.show()


# In[28]:


import matplotlib.pyplot as plt

# Set up a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Prediction results for training dataset
L_train = len(x_train)
axes[0, 0].plot(range(L_train), y_train, color='red')
axes[0, 0].plot(range(L_train), trainpreds, color='blue')
axes[0, 0].set_ylabel("CO2 Solubility (mol/kg)", fontsize=10)
axes[0, 0].set_title("Training Dataset")
axes[0, 0].legend(['actual', 'predicted'])

# Prediction results for Testing dataset
L_test = len(x_test)
axes[0, 1].plot(range(L_test), y_test, color='red')
axes[0, 1].plot(range(L_test), testpreds, color='blue')
axes[0, 1].set_ylabel("CO2 Solubility (mol/kg)", fontsize=10)
axes[0, 1].set_title("Validation Dataset")
axes[0, 1].legend(['actual', 'predicted'])



# Adjust layout for better spacing
fig.tight_layout()

# Show the plot
plt.show()


# In[47]:


# Getting the last row of data as a sample input for a single block
last_row_features = Train_data.tail(999)
last_row_features


# In[48]:


import time
from tqdm import tqdm
# Grid blocks in a 2D space
maximum_grid_block_x = 10
maximum_grid_block_y = 10

def process_block(i, j):

    # metrics and prediction of all data
    single_predict = best_lssvm_model.predict(last_row_features)
    return single_predict

# Start the timer
start_time = time.time()

# Using tqdm for the overall progress bar
total_blocks = maximum_grid_block_x * maximum_grid_block_y
with tqdm(total=total_blocks, desc='Processing Blocks') as pbar:
    for i in range(1, maximum_grid_block_x + 1):
        for j in range(1, maximum_grid_block_y + 1):
            process_block(i, j)
            # Update progress bar
            pbar.update(1)  # Update for each processed block
            print(f'Solution for grid block X = {i} and Y = {j}')

# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f'Time taken for the loop to execute: {elapsed_time:.6f} seconds')



# In[ ]:





# In[ ]:




