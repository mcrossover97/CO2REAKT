#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Pandas is used for data manipulation
import pandas as pd# Read in data and display first 5 rows
df = pd.read_csv('../../datasets/Normilized_dataset.csv')


# Use numpy to convert to arrays
import numpy as np
import matplotlib. pyplot as plt
from matplotlib import rcParams
import matplotlib. pyplot as plt 
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


# In[3]:


print('The shape of our features is:', df.shape)


# In[4]:


# Descriptive statistics for each column
df.describe()


# In[5]:


from matplotlib import rcParams
import matplotlib. pyplot as plt 
rcParams['figure.figsize'] = 8, 8


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
feature_data= df.drop('CO2 Solubility (mol/kg)', axis = 1)


# In[10]:


feature_data.shape


# In[11]:


# Saving feature names for later use
feature_list = list(feature_data.columns)


# In[12]:


feature_list


# In[13]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split


# In[14]:


# 20% for validation
x_train, x_val, y_train, y_val = train_test_split(feature_data, label_data, test_size=0.2, random_state=10) # 0.20 x 1 = 0.20


# In[15]:


# 10% for testing
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.125, random_state=101) # 0.125 x 0.8 = 0.1


# In[16]:


print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Validating Features Shape:', x_val.shape)
print('Validating Labels Shape:', y_val.shape)
print('testing Features Shape:', x_test.shape)
print('testing Labels Shape:', y_test.shape)


# In[17]:


import xgboost as xgb


# In[18]:


# Step Convert data to XGBoost DMatrix format
dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_val, label=y_val)
dtest = xgb.DMatrix(x_test, label=y_test)
dall = xgb.DMatrix(feature_data, label=label_data)


# In[19]:


dtrain


# In[20]:


dval


# In[28]:


# Define hyperparameters
params = {
    
    'learning_rate': [1e-1,1e-2,1e-3,1e-4],
    'n_estimators': [10,20,50,100],
    'max_depth': [5,10,20,100],
    'min_child_weight': [1,3,5,10],
    'gamma': [0 ,5,10,20,50],
}


# In[29]:


# Lists to store training loss results
train_results = []
evals_result = {}


# In[30]:


# Define a callback function to record training and evaluation results
def callback(epoch, _):
    train_results.append(model.eval(dtrain))
    if epoch % 10 == 0:  # Record evaluation results every 10 rounds
        evals_result[f'epoch_{epoch}'] = model.eval(dval)


# In[31]:


#Train the XGBoost model with early stopping based on validation performance
evals = [(dtrain, 'train'), (dval, 'validation')]


# In[32]:


# Create the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror')


# In[35]:


# Perform grid search 
grid_search = GridSearchCV(estimator = model, param_grid=params, cv=5, scoring='neg_mean_squared_error', verbose=1)


# In[36]:


with tqdm(total=len(params['learning_rate'])*len(params['n_estimators']),desc='Grid Search bar') as pbar :
    grid_search.fit(x_train,y_train)
    pbar.update(1)
    


# In[37]:


# Print the best hyperparameter values and corresponding RMSE score
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best RMSE Score: ", grid_search.best_score_)


# In[38]:


# Train the XGBoost model with early stopping
model = xgb.train(grid_search.best_params_,dtrain, num_boost_round=1000,  evals=[(dtrain, 'train'), (dval, 'validation')],
                  early_stopping_rounds=20, evals_result=evals_result, verbose_eval=True)


# In[39]:


# Save the trained model to a file
model.save_model('XGBoost_CV_5.json')  # You can use a .json file format


# In[40]:


#model = xgb.XGBClassifier()  # Make sure to create an instance of your model
model.load_model('XGBoost_CV_5.json')


# In[41]:


# Extract training and validation metrics from the results
train_loss = evals_result['train']['rmse']
val_loss = evals_result['validation']['rmse']


# In[42]:


# Plot the training and validation loss graph
fig, ax = plt.subplots(figsize=(8, 8))
epochs = range(0, len(train_loss))
ax.plot(epochs, train_loss, 'bo', label='Training Loss')
ax.plot(epochs, val_loss, 'orange', label='Validation Loss')

# Set the y-axis label to "Loss"
ax.set_xlabel('Boosting Rounds')
ax.set_ylabel('Loss', color='blue')
ax.tick_params(axis='y', labelcolor='blue')

# Add a legend to the plot
ax.legend()

# Save the figure
plt.savefig('scatter_plot.png', dpi=300)

# Show the plot
plt.show()


# In[43]:


# metrics and prediction of test data
testpreds = model.predict(dtest)


# In[44]:


print('test mae = ',mean_absolute_error(y_test, testpreds)) # test


# In[45]:


print('test mse = ',mean_squared_error(y_test, testpreds)) # test


# In[46]:


acc_test = r2_score(y_test, testpreds)
acc_test


# In[47]:


# metrics and prediction of test data
trainpreds = model.predict(dtrain)


# In[48]:


print('train mae = ',mean_absolute_error(y_train, trainpreds)) # test


# In[49]:


print('train mse = ',mean_squared_error(y_train, trainpreds)) # test


# In[50]:


acc_train = r2_score(y_train, trainpreds)
acc_train


# In[51]:


# metrics and prediction of test data
valpreds = model.predict(dval)


# In[52]:


print('validation mae = ',mean_absolute_error(y_val, valpreds)) # test


# In[53]:


print('validation mse = ',mean_squared_error(y_val, valpreds)) # test


# In[54]:


acc_val = r2_score(y_val, valpreds)
acc_val


# In[55]:


# metrics and prediction of test data
allpreds = model.predict(dall)


# In[56]:


print('All mae = ',mean_absolute_error(label_data, allpreds)) # All


# In[57]:


print('All mse = ',mean_squared_error(label_data, allpreds)) # All


# In[58]:


acc_all = r2_score(label_data, allpreds)
acc_all


# In[59]:


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
sns.regplot(x=y_train, y=trainpreds, scatter_kws={'alpha': 0.5, 'color': 'purple'}, line_kws={'color': 'purple'}, ci=90, ax=axes[0, 0], label="Train Regression Line")
sns.scatterplot(x=y_train, y=trainpreds, color='blue', alpha=0.5, ax=axes[0, 0], label="Train Original Data")
axes[0, 0].set_xlabel("Actual CO2 Solubility (mol/kg)", fontsize=10)
axes[0, 0].set_ylabel("Predicted CO2 Solubility (mol/kg)", fontsize=10)
axes[0, 0].set_title("Training Results")
axes[0, 0].legend(loc='upper left')
# Add R-squared to the bottom right
r_squared_train = r2_score(y_train, trainpreds)
axes[0, 0].text(0.95, 0.05, f'R-squared = {r_squared_train:.3f}', ha='right', va='bottom', transform=axes[0, 0].transAxes)

# Plot for Validation Results
sns.regplot(x=y_val, y=valpreds, scatter_kws={'alpha': 0.5, 'color': 'green'}, line_kws={'color': 'green'}, ci=90, ax=axes[0, 1], label="Validation Regression Line")
sns.scatterplot(x=y_val, y=valpreds, color='orange', alpha=0.5, ax=axes[0, 1], label="Validation Original Data")
axes[0, 1].set_xlabel("Actual CO2 Solubility (mol/kg)", fontsize=10)
axes[0, 1].set_ylabel("Predicted CO2 Solubility (mol/kg)", fontsize=10)
axes[0, 1].set_title("Validation Results")
axes[0, 1].legend(loc='upper left')
# Add R-squared to the bottom right
r_squared_val = r2_score(y_val, valpreds)
axes[0, 1].text(0.95, 0.05, f'R-squared = {r_squared_val:.3f}', ha='right', va='bottom', transform=axes[0, 1].transAxes)

# Plot for Testing Results
sns.regplot(x=y_test, y=testpreds, scatter_kws={'alpha': 0.5, 'color': 'red'}, line_kws={'color': 'red'}, ci=90, ax=axes[1, 0], label="Testing Regression Line")
sns.scatterplot(x=y_test, y=testpreds, color='black', alpha=0.5, ax=axes[1, 0], label="Testing Original Data")
axes[1, 0].set_xlabel("Actual CO2 Solubility (mol/kg)", fontsize=10)
axes[1, 0].set_ylabel("Predicted CO2 Solubility (mol/kg)", fontsize=10)
axes[1, 0].set_title("Testing Results")
axes[1, 0].legend(loc='upper left')
# Add R-squared to the bottom right
r_squared_test = r2_score(y_test, testpreds)
axes[1, 0].text(0.95, 0.05, f'R-squared = {r_squared_test:.3f}', ha='right', va='bottom', transform=axes[1, 0].transAxes)

# Plot for Blind Results
sns.regplot(x=label_data.flatten(), y=allpreds.flatten(), scatter_kws={'alpha': 0.5, 'color': 'purple'}, line_kws={'color': 'purple'}, ci=90, ax=axes[1, 1], label="All Regression Line")
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


# In[60]:


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

# Prediction results for validation dataset
L_val = len(x_val)
axes[0, 1].plot(range(L_val), y_val, color='red')
axes[0, 1].plot(range(L_val), valpreds, color='blue')
axes[0, 1].set_ylabel("CO2 Solubility (mol/kg)", fontsize=10)
axes[0, 1].set_title("Validation Dataset")
axes[0, 1].legend(['actual', 'predicted'])

# Prediction results for testing dataset
L_test = len(x_test)
axes[1, 0].plot(range(L_test), y_test, color='red')
axes[1, 0].plot(range(L_test), testpreds, color='blue')
axes[1, 0].set_ylabel("CO2 Solubility (mol/kg)", fontsize=10)
axes[1, 0].set_title("Testing Dataset")
axes[1, 0].legend(['actual', 'predicted'])

# Prediction results for Blind dataset
L_All = min(len(label_data.flatten()), len(allpreds.flatten()))
axes[1, 1].plot(range(L_All), label_data.flatten()[:L_All], color='red')
axes[1, 1].plot(range(L_All), allpreds.flatten()[:L_All], color='blue')
axes[1, 1].set_ylabel("CO2 Solubility (mol/kg)", fontsize=10)
axes[1, 1].set_title("All Dataset")
axes[1, 1].legend(['actual', 'predicted'])

# Adjust layout for better spacing
fig.tight_layout()

# Show the plot
plt.show()


# In[55]:


# Getting the last row of data as a sample input for a single block
last_row_features = feature_data.tail(1)
last_row_features


# In[56]:


# Getting the last row of data as a sample input for a single block
last_row_label = label_data[-1]
last_row_label


# In[57]:


dlast_row = xgb.DMatrix(last_row_features, label=last_row_label)


# In[58]:


import time
from tqdm import tqdm
# Grid blocks in a 2D space
maximum_grid_block_x = 100
maximum_grid_block_y = 100

def process_block(i, j):

    # metrics and prediction of all data
    single_predict = model.predict(dlast_row)
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




