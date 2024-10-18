#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Pandas is used for data manipulation
import pandas as pd# Read in data and display first 5 rows
df = pd.read_csv('../../datasets/Normilized_dataset.csv')

# Use numpy to convert to arrays
import numpy as np

# Data preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# 
from sklearn.metrics import r2_score
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error , mean_squared_error


# In[7]:


print('The shape of our features is:', df.shape)


# In[8]:


# Descriptive statistics for each column
df.describe()


# In[9]:


df.info()


# In[10]:


from matplotlib import rcParams
import matplotlib. pyplot as plt 
rcParams['figure.figsize'] = 8, 8


# In[11]:


# view normalized data 
print(df)


# In[12]:


# Labels are the values we want to predict
label_data = np.array(df['CO2 Solubility (mol/kg)'])


# In[13]:


label_data.shape


# In[14]:


# Remove the labels from the features
# axis 1 refers to the columns
feature_data= df.drop('CO2 Solubility (mol/kg)', axis = 1)


# In[15]:


feature_data.shape


# In[16]:


# Saving feature names for later use
feature_list = list(feature_data.columns)


# In[17]:


feature_list


# In[18]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split


# In[19]:


# 20% for validation
x_train, x_val, y_train, y_val = train_test_split(feature_data, label_data, test_size=0.2, random_state=10) # 0.20 x 1 = 0.20




# In[20]:


# 10% for testing
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.125, random_state=101) # 0.125 x 0.8 = 0.1


# In[21]:


x_train


# In[22]:


y_train


# In[23]:


x_val


# In[24]:


y_val


# In[25]:


x_test


# In[26]:


y_test.shape


# In[27]:


print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Validating Features Shape:', x_val.shape)
print('Validating Labels Shape:', y_val.shape)
print('testing Features Shape:', x_test.shape)
print('testing Labels Shape:', y_test.shape)


# In[28]:


# Setup the parameter grid for GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200,500],
    'max_depth': [5,10, 20, 30,50],
    'min_samples_split': [2,5,10,20],
    'min_samples_leaf': [1, 2, 5,10],
    'max_features': [1, 2, 5, 10,20]
}


# In[29]:


# Step 4: Initialize the RandomForestRegressor
model = RandomForestRegressor(random_state=42)


# In[30]:


# Step 5: Initialize GridSearchCV with 5-fold cross-validation
model = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, 
                           scoring='neg_mean_squared_error', n_jobs=-1, verbose=2,error_score='raise')


# In[31]:


# Step 6: Fit GridSearchCV to the data
model.fit(x_train, y_train)


# In[26]:


# Step 7: Get the best parameters and the best score
best_params = model.best_params_
best_score = -model.best_score_  # Convert from negative MSE to positive MSE

print(f"Best parameters found: {best_params}")
print(f"Best Mean Squared Error from cross-validation: {best_score}")


# In[28]:


# Setup the parameter for the best model (best hyperparamter were chosen because I don't want to run CV again!)

param_grid = {
    'n_estimators': [200],
    'max_depth': [20],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': [5]
}


# In[29]:


# Step 4: Initialize the RandomForestRegressor
model = RandomForestRegressor(random_state=42)


# In[30]:


# Step 5: Initialize GridSearchCV with 5-fold cross-validation
model = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, 
                           scoring='neg_mean_squared_error', n_jobs=-1, verbose=2,error_score='raise')


# In[31]:


# Step 6: Fit GridSearchCV to the data
model.fit(x_train, y_train)


# In[ ]:





# In[32]:


import pickle


# In[33]:


# Save the model to a file
with open('random_forest_regressor_CV_4.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[34]:


# Save the model to a file
with open('random_forest_regressor_CV_4.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[35]:


x_train.shape


# In[36]:


y_train.shape


# In[37]:


# metrics and prediction of test data
testpreds = model.predict(x_test)


# In[38]:


print('test MAE = ',mean_absolute_error(y_test, testpreds)) # test
print('test MSE = ',mean_squared_error(y_test, testpreds)) # test


# In[39]:


acc_test = r2_score(y_test, testpreds)
acc_test


# In[40]:


# metrics and prediction of train data
trainpreds = model.predict(x_train)


# In[41]:


print('train MAE = ',mean_absolute_error(y_train, trainpreds)) # train
print('test MSE = ',mean_squared_error(y_train, trainpreds)) # train


# In[42]:


acc_train = r2_score(y_train, trainpreds)
acc_train


# In[43]:


# metrics and prediction of train data
valpreds = model.predict(x_val)


# In[44]:


print('train MAE = ',mean_absolute_error(y_val, valpreds)) # val
print('train MSE = ',mean_squared_error(y_val, valpreds)) # val


# In[45]:


acc_val = r2_score(y_val, valpreds)
acc_val


# In[46]:


# metrics and prediction of blind data
allpreds = model.predict(feature_data)


# In[47]:


print('All MAE = ',mean_absolute_error(label_data, allpreds)) # val
print('All MSE = ',mean_squared_error(label_data, allpreds)) # val


# In[48]:


acc_all = r2_score(label_data, allpreds)
acc_all


# In[49]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Assuming 'y_train', 'trainpreds', 'y_val', and 'valpreds' are defined

# If 'y_train' and 'trainpreds' are 2D, you can use the first column
y_train = y_train.flatten() if len(y_train.shape) > 1 else y_train
trainpreds = trainpreds.flatten() if len(trainpreds.shape) > 1 else trainpreds

# If 'y_val' and 'valpreds' are 2D, you can use the first column
y_val = y_val.flatten() if len(y_val.shape) > 1 else y_val
valpreds = valpreds.flatten() if len(valpreds.shape) > 1 else valpreds

# Flatten y_test and testpreds if they are 2D
y_test = y_test.flatten() if len(y_test.shape) > 1 else y_test
testpreds = testpreds.flatten() if len(testpreds.shape) > 1 else testpreds



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

# Plot for Validation Results
sns.regplot(x=y_val, y=valpreds, scatter_kws={'alpha': 0.5, 'color': 'green'}, line_kws={'color': 'green'}, ci=0, ax=axes[0, 1], label="Validation Regression Line")
sns.scatterplot(x=y_val, y=valpreds, color='orange', alpha=0.5, ax=axes[0, 1], label="Validation Original Data")
axes[0, 1].set_xlabel("Actual CO2 Solubility (mol/kg)", fontsize=10)
axes[0, 1].set_ylabel("Predicted CO2 Solubility (mol/kg)", fontsize=10)
axes[0, 1].set_title("Validation Results")
axes[0, 1].legend(loc='upper left')
# Add R-squared to the bottom right
r_squared_val = r2_score(y_val, valpreds)
axes[0, 1].text(0.95, 0.05, f'R-squared = {r_squared_val:.3f}', ha='right', va='bottom', transform=axes[0, 1].transAxes)

# Plot for Testing Results
sns.regplot(x=y_test, y=testpreds, scatter_kws={'alpha': 0.5, 'color': 'red'}, line_kws={'color': 'red'}, ci=0, ax=axes[1, 0], label="Testing Regression Line")
sns.scatterplot(x=y_test, y=testpreds, color='black', alpha=0.5, ax=axes[1, 0], label="Testing Original Data")
axes[1, 0].set_xlabel("Actual CO2 Solubility (mol/kg)", fontsize=10)
axes[1, 0].set_ylabel("Predicted CO2 Solubility (mol/kg)", fontsize=10)
axes[1, 0].set_title("Testing Results")
axes[1, 0].legend(loc='upper left')
# Add R-squared to the bottom right
r_squared_test = r2_score(y_test, testpreds)
axes[1, 0].text(0.95, 0.05, f'R-squared = {r_squared_test:.3f}', ha='right', va='bottom', transform=axes[1, 0].transAxes)

# Plot for Blind Results
sns.regplot(x=label_data.flatten(), y=allpreds.flatten(), scatter_kws={'alpha': 0.5, 'color': 'purple'}, line_kws={'color': 'purple'}, ci=0, ax=axes[1, 1], label="All Regression Line")
sns.scatterplot(x=label_data.flatten(), y=allpreds.flatten(), color='red', alpha=0.5, ax=axes[1, 1], label="All Original Data")
axes[1, 1].set_xlabel("Actual CO2 Solubility (mol/kg)", fontsize=10)
axes[1, 1].set_xlabel("Actual CO2 Solubility (mol/kg)", fontsize=10)
axes[1, 1].set_ylabel("Predicted CO2 Solubility (mol/kg)", fontsize=10)
axes[1, 1].set_title("All Results")
axes[1, 1].legend(loc='upper left')
# Add R-squared to the bottom right
r_squared_blind = r2_score(label_data, allpreds)
axes[1, 1].text(0.95, 0.05, f'R-squared = {r_squared_blind:.3f}', ha='right', va='bottom', transform=axes[1, 1].transAxes)




# Set the DPI (dots per inch)
plt.figure(dpi=300)


# show the plot
plt.show()


# In[50]:


import matplotlib.pyplot as plt

# Set up a 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Prediction results for training dataset
L_train = len(x_train)
axes[0, 0].plot(range(L_train), y_train, color='red')
axes[0, 0].plot(range(L_train), trainpreds, color='blue')
axes[0, 0].set_ylabel("QL", fontsize=10)
axes[0, 0].set_title("Training Dataset")
axes[0, 0].legend(['actual', 'predicted'])

# Prediction results for validation dataset
L_val = len(x_val)
axes[0, 1].plot(range(L_val), y_val, color='red')
axes[0, 1].plot(range(L_val), valpreds, color='blue')
axes[0, 1].set_ylabel("QL", fontsize=10)
axes[0, 1].set_title("Validation Dataset")
axes[0, 1].legend(['actual', 'predicted'])

# Prediction results for testing dataset
L_test = len(x_test)
axes[1, 0].plot(range(L_test), y_test, color='red')
axes[1, 0].plot(range(L_test), testpreds, color='blue')
axes[1, 0].set_ylabel("QL", fontsize=10)
axes[1, 0].set_title("Testing Dataset")
axes[1, 0].legend(['actual', 'predicted'])

# Prediction results for All dataset
L_All = len(feature_data)
axes[1, 1].plot(range(L_All), label_data, color='red')
axes[1, 1].plot(range(L_All), allpreds, color='blue')
axes[1, 1].set_ylabel("QL", fontsize=10)
axes[1, 1].set_title("All Dataset")
axes[1, 1].legend(['actual', 'predicted'])

# Adjust layout for better spacing
fig.tight_layout()

# Show the plot
plt.show()


# In[51]:


# Getting the last row of data as a sample input for a single block
last_row = feature_data.tail(999)


# In[52]:


import time
from tqdm import tqdm
# Grid blocks in a 2D space
maximum_grid_block_x = 10
maximum_grid_block_y = 10

def process_block(i, j):

    # metrics and prediction of all data
    single_predict = model.predict(last_row)
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





# In[ ]:





# In[ ]:




