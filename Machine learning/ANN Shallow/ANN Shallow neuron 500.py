#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Pandas is used for data manipulation
import pandas as pd# Read in data and display first 5 rows
df = pd.read_csv('../../datasets/Normilized_dataset.csv')

# Use numpy to convert to arrays
import numpy as np

# Data preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
# 
from sklearn.metrics import r2_score
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error , mean_squared_error


# In[3]:


print('The shape of our features is:', df.shape)


# In[4]:


# Descriptive statistics for each column
df.describe()


# In[5]:


df.info()


# In[6]:


from matplotlib import rcParams
import matplotlib. pyplot as plt 
rcParams['figure.figsize'] = 8, 8


# In[7]:


# view normalized data 
print(df)


# In[8]:


# Labels are the values we want to predict
label_data = np.array(df['CO2 Solubility (mol/kg)'])


# In[9]:


label_data.shape


# In[10]:


# Remove the labels from the features
# axis 1 refers to the columns
feature_data= df.drop('CO2 Solubility (mol/kg)', axis = 1)


# In[11]:


feature_data.shape


# In[12]:


# Saving feature names for later use
feature_list = list(feature_data.columns)


# In[13]:


feature_list


# In[14]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split


# In[15]:


# 20% for validation
x_train, x_val, y_train, y_val = train_test_split(feature_data, label_data, test_size=0.2, random_state=10) # 0.20 x 1 = 0.20




# In[16]:


# 10% for testing
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.125, random_state=101) # 0.125 x 0.8 = 0.1


# In[17]:


x_train


# In[18]:


y_train


# In[19]:


x_val


# In[30]:


y_val


# In[31]:


x_test


# In[32]:


y_test.shape


# In[33]:


print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Validating Features Shape:', x_val.shape)
print('Validating Labels Shape:', y_val.shape)
print('testing Features Shape:', x_test.shape)
print('testing Labels Shape:', y_test.shape)


# In[34]:


# Neural Net modules
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import backend
from keras import optimizers
from keras.layers import LSTM
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.utils import plot_model
import keras_tuner
import keras



# In[35]:


# build a Shallow model with neurons 500


# In[36]:


backend.clear_session()
model = Sequential()
model.add(Dense(500, input_shape=(x_train.shape[1],),  activation="relu")) # (features,)
model.add(Dense(1, activation='linear')) # output node
model.summary() # see what your model looks like
# compile the model
model.compile(optimizer= optimizers.Adam(learning_rate=1e-4) ,loss='mse', metrics=['mae'])


# In[37]:


# early stopping callback
es = EarlyStopping(monitor='val_loss',mode='min',patience=20,restore_best_weights = True)


# In[38]:


#from tensorflow.keras.utils import plot_model


plot_model(model, show_shapes=True, show_layer_names=True)



# In[39]:


# fit the model!
# attach it to a new variable called 'history' in case
# to look at the learning curves
history = model.fit(x_train, y_train,
                    validation_data = (x_val, y_val),
                    callbacks=[es],
                    epochs=10000,
                    batch_size=16)


# In[40]:


x_train.shape


# In[41]:


y_train.shape


# In[42]:


# let's see the training and validation accuracy by epoch
history_dict = history.history
loss_values = history_dict['loss'] # you can change this
val_loss_values = history_dict['val_loss'] # you can also change this
epochs = range(1, len(loss_values) + 1) # range of X (no. of epochs)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_process.jpg',dpi = 300)
plt.show()


# In[43]:


# Saving the model , so we ensure that our model doesn't waste
model.save('Model_ANN_500.h5')


# In[44]:


# loading the model
model = tf.keras.models.load_model('Model_ANN_500.h5')


# In[45]:


# metrics and prediction of test data
testpreds = model.predict(x_test)


# In[46]:


print('test MAE = ',mean_absolute_error(y_test, testpreds)) # test
print('test MSE = ',mean_squared_error(y_test, testpreds)) # test


# In[47]:


acc_test = r2_score(y_test, testpreds)
acc_test


# In[48]:


# metrics and prediction of train data
trainpreds = model.predict(x_train)


# In[49]:


print('train MAE = ',mean_absolute_error(y_train, trainpreds)) # train
print('test MSE = ',mean_squared_error(y_train, trainpreds)) # train


# In[50]:


acc_train = r2_score(y_train, trainpreds)
acc_train


# In[51]:


# metrics and prediction of train data
valpreds = model.predict(x_val)


# In[52]:


print('train MAE = ',mean_absolute_error(y_val, valpreds)) # val
print('train MSE = ',mean_squared_error(y_val, valpreds)) # val


# In[53]:


acc_val = r2_score(y_val, valpreds)
acc_val


# In[54]:


# metrics and prediction of blind data
allpreds = model.predict(feature_data)


# In[55]:


print('All MAE = ',mean_absolute_error(label_data, allpreds)) # val
print('All MSE = ',mean_squared_error(label_data, allpreds)) # val


# In[56]:


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

# Prediction results for All dataset
L_All = len(feature_data)
axes[1, 1].plot(range(L_All), label_data, color='red')
axes[1, 1].plot(range(L_All), allpreds, color='blue')
axes[1, 1].set_ylabel("CO2 Solubility (mol/kg)", fontsize=10)
axes[1, 1].set_title("All Dataset")
axes[1, 1].legend(['actual', 'predicted'])

# Adjust layout for better spacing
fig.tight_layout()

# Show the plot
plt.show()


# In[37]:


# Getting the last row of data as a sample input for a single block
last_row = feature_data.tail(1)


import time
from tqdm import tqdm
# Grid blocks in a 2D space
maximum_grid_block_x = 100
maximum_grid_block_y = 100

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




