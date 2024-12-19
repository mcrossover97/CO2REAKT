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


#importing Keras libraries
import keras
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.utils import plot_model
from keras import optimizers
from keras import backend
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


# In[4]:


print('The shape of our features is:', df.shape)


# In[5]:


# Descriptive statistics for each column
df.describe()


# In[6]:


df.info()


# In[7]:


from matplotlib import rcParams
import matplotlib. pyplot as plt 
rcParams['figure.figsize'] = 8, 8


# In[8]:


# Labels are the values we want to predict
label_data = np.array(df['CO2 Solubility (mol/kg)'])


# In[9]:


# Remove the labels from the features
# axis 1 refers to the columns
feature_data= df.drop('CO2 Solubility (mol/kg)', axis = 1)


# In[10]:


# Saving feature names for later use
feature_list = list(feature_data.columns)


# In[11]:


feature_list


# In[12]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split


# In[13]:


# 20% for validation
x_train, x_val, y_train, y_val = train_test_split(feature_data, label_data, test_size=0.2, random_state=10) # 0.20 x 1 = 0.20




# In[14]:


# 10% for testing
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.125, random_state=101) # 0.125 x 0.8 = 0.1


# In[15]:


x_train


# In[16]:


y_train


# In[17]:


x_val


# In[18]:


y_val


# In[19]:


x_test


# In[20]:


y_test.shape


# In[21]:


print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Validating Features Shape:', x_val.shape)
print('Validating Labels Shape:', y_val.shape)
print('testing Features Shape:', x_test.shape)
print('testing Labels Shape:', y_test.shape)


# In[23]:


# Building the CNN model

# build the model
backend.clear_session()

model = Sequential()

# Add a 1D convolutional layer with 32 filters, kernel size 6, and activation function 'relu'
model.add(Conv1D(32, kernel_size=6, activation='relu', input_shape=(6, 1)))

# Add a max pooling layer with pool size 1
model.add(MaxPooling1D(pool_size=1))

# Flatten the output to a 1D array
model.add(Flatten())

# Add a dense layer with 64 neurons and activation function 'relu'
model.add(Dense(256, activation='relu'))

# Add the output layer with 1 neuron and activation function 'linear'
model.add(Dense(1, activation='linear'))
model.summary() # see what your model looks like
# Compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# compile the model
model.compile(optimizer= optimizers.Adam(learning_rate=1e-2) ,loss='mse', metrics=['mae'])


# In[24]:


# early stopping callback
es = EarlyStopping(monitor='val_loss',mode='min',patience=20,restore_best_weights = True)


# In[25]:


#from tensorflow.keras.utils import plot_model
plot_model(model, to_file='CNN_without_Tuning.png', show_shapes=True, show_layer_names=True)


# In[26]:


# fit the model!
# attach it to a new variable called 'history' in case
# to look at the learning curves
history = model.fit(x_train, y_train,
                    validation_data = (x_val, y_val),
                    callbacks=[es],
                    epochs=1000,
                    batch_size=64)


# In[27]:


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


# In[28]:


# Saving the model , so we ensure that our model doesn't waste
model.save('CNN_kernel_size_6_pool_size_1_Lr_1e-2_BS_64_Max_pool_Dense_256')


# In[29]:


# loading the model
model = tf.keras.models.load_model('CNN_kernel_size_6_pool_size_1_Lr_1e-2_BS_64_Max_pool_Dense_256')


# In[30]:


# metrics and prediction of test data
testpreds = model.predict(x_test)


# In[31]:


print('test mae = ',mean_absolute_error(y_test, testpreds)) # test


# In[32]:


acc_test = r2_score(y_test, testpreds)
acc_test


# In[33]:


# metrics and prediction of train data
trainpreds = model.predict(x_train)


# In[34]:


print('train mae = ',mean_absolute_error(y_train, trainpreds)) # train


# In[35]:


acc_train = r2_score(y_train, trainpreds)
acc_train


# In[36]:


# metrics and prediction of train data
valpreds = model.predict(x_val)


# In[37]:


print('train mae = ',mean_absolute_error(y_val, valpreds)) # train


# In[38]:


acc_train = r2_score(y_train, trainpreds)
acc_train


# In[39]:


# metrics and prediction of blind data
allpreds = model.predict(feature_data)


# In[40]:


print('All MAE = ',mean_absolute_error(label_data, allpreds)) # val
print('All MSE = ',mean_squared_error(label_data, allpreds)) # val


# In[41]:


acc_all = r2_score(label_data, allpreds)
acc_all


# In[44]:


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

# tight layout
fig.tight_layout()
# show the plot
plt.show()


# In[43]:


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


# In[33]:


# Getting the last row of data as a sample input for a single block
last_row = feature_data.tail(999)
last_row


# In[34]:


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



