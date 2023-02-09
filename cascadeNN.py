import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.callbacks import ModelCheckpoint
import seaborn as sns
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import keras
from keras.layers import Input, LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, Lambda
from keras.models import Model
import tensorflow as tf

# df = pd.read_csv('./Dataset/puredataset/CascadeNN/alldataset.csv')
# counthr = df['HR'].count()
# countdbp = df['DBP'].count()
# countsbp = df['SBP'].count()
# print(f'number of hr:{counthr}, DBP:{countdbp},SBP: {countsbp}')

def get_model(u_conv1_1=64,u_conv2_1=64,u_LSTM = 128):

    print(f'Make Model with units of Conv1-1:{u_conv1_1} -Conv2-1:{u_conv2_1} - LSTM : {u_LSTM}')

    combi_input = Input(shape=(2,))
    input1 = Lambda(lambda x: tf.expand_dims(x[:,0],-1))(combi_input)
    input2 = Lambda(lambda x: tf.expand_dims(x[:,1],-1))(combi_input)

    input1R = Lambda(lambda x: tf.reshape(x,[-1,1,1]))(input1)
    input2R = Lambda(lambda x: tf.reshape(x,[-1,1,1]))(input2)

    # Define the first branch of the HR signal
    conv1_1 = Conv1D(filters=u_conv1_1, kernel_size=1,input_shape=(1, 1), activation='relu')(input1R)
    pool1_1 = MaxPooling1D(pool_size=1)(conv1_1)

    conv1_2 = Conv1D(filters=u_conv1_1, kernel_size=1, activation='relu')(input1R)
    pool1_2 = MaxPooling1D(pool_size=1)(conv1_2)

    concat1 = Concatenate()([pool1_1, pool1_2])

    conv1_3 = Conv1D(filters=u_conv2_1, kernel_size=1, activation='relu')(concat1)
    pool1_3 = MaxPooling1D(pool_size=1)(conv1_3)

    conv1_4 = Conv1D(filters=u_conv2_1, kernel_size=1, activation='relu')(concat1)
    pool1_4 = MaxPooling1D(pool_size=1)(conv1_4)

    concat2 = Concatenate()([pool1_3, pool1_4])

    # Define the second branch of the PPG signal
    conv2_1 = Conv1D(filters=u_conv1_1, kernel_size=1, activation='relu')(input2R)
    pool2_1 = MaxPooling1D(pool_size=1)(conv2_1)

    conv2_2 = Conv1D(filters=u_conv1_1, kernel_size=1, activation='relu')(input2R)
    pool2_2 = MaxPooling1D(pool_size=1)(conv2_2)

    concat3 = Concatenate()([pool2_1, pool2_2])

    conv2_3 = Conv1D(filters=u_conv2_1, kernel_size=1, activation='relu')(concat3)
    pool2_3 = MaxPooling1D(pool_size=1)(conv2_3)

    conv2_4 = Conv1D(filters=u_conv2_1, kernel_size=1, activation='relu')(concat3)
    pool2_4 = MaxPooling1D(pool_size=1)(conv2_4)

    concat4 = Concatenate()([pool2_3, pool2_4])

    # Concatenate the outputs of the two branches
    concat5 = Concatenate()([concat2, concat4])

    # Define the LSTM layer
    lstm = LSTM(units=u_LSTM)(concat5)
    flatten = Flatten()(lstm)

    # dense = Dense(units=64, activation='relu')(flatten)
    # Dense layer with 2 outputs
    outputs = Dense(2, activation='linear')(flatten)

    # Define the model
    model = keras.models.Model(inputs=combi_input, outputs=outputs)

    # Compile the model
    opt = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(loss='mean_squared_error', optimizer=opt)

    return model

def plot_all(y,y_hat):
  for i in range(2):
    if i==0:
      _title = 'DBP'
    else:
      _title = 'SBP'

    # plot Desired values and Predict values
    plt.figure()
    plt.plot(y_hat[:,i],c='r')
    plt.plot(y[:,i],c='b')
    plt.show()

    # plot regression plot 
    plt.figure()
    sns.regplot(x=y[:,i],y=y_hat[:,i], scatter_kws={'s': 2})
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Regression Plot of True Values vs. Predicted Values({_title})")
    plt.show()

    # Plot error histogram
    error = y [:,i]- y_hat[:,i]

    plt.hist(error, bins=20, rwidth=0.8)
    plt.xlabel("Error(mmHg)")
    plt.ylabel("Frequency")

    plt.title(f"Histogram of Error ({_title})")
    plt.show()

def training(X_train, X_val,y_train,y_val):
  # num_features = 2

  model = get_model()

  checkpoint = ModelCheckpoint("./best_model.h5", monitor='val_loss', save_best_only=True, mode='min')

  history = model.fit(X_train,
                      y_train, #batch_size = 32,
                      validation_data=(X_val, y_val),
                      epochs=100)

  trainPredict = model.predict(X_train)
  validPredict = model.predict(X_val)
  testPredict = model.predict(X_test)
  print(f'shape of train predict{trainPredict.shape}')

  trainScore = math.sqrt(mean_squared_error(y_train[:,0], trainPredict[:,0]))
  testScore = math.sqrt(mean_squared_error(y_test[:,0], testPredict[:,0]))

  print( 'Train Score DBP: %.2f RMSE' % (trainScore))
  print( 'Test Score DBP: %.2f RMSE' % (testScore))

  trainScore = math.sqrt(mean_squared_error(y_train[:,1], trainPredict[:,1]))
  testScore = math.sqrt(mean_squared_error(y_test[:,1], testPredict[:,1]))
  print( 'Train Score SBP: %.2f RMSE' % (trainScore))
  print( 'Test Score SBP: %.2f RMSE' % (testScore))

  plt.figure()
  print(history.history.keys())
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train','Val'], loc='upper left')
  # plt.savefig(f'./Plots/LSTM/{patient}_loss.png')
  plt.show()
  return model

def run_gridsearch(xtrain,ytrain, param_grid):

  model = KerasRegressor(build_fn=get_model, epochs=70, batch_size=32, verbose=0)

  kfold = KFold(n_splits=5, shuffle=True)

  grid = GridSearchCV(estimator=model, param_grid=param_grid,# n_jobs=-1,
                      cv=kfold, verbose=1,
                      return_train_score=True)
  
  grid_result = grid.fit(xtrain, ytrain)
  print(f"best_estimator_: {grid_result.best_estimator_}")
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

df = pd.read_csv('./Dataset/Regression/3402408.csv').set_index('Time')
# print(df.head())
df = df.dropna(how='all')
x_final = np.arange(0, 60,.1)
for i in x_final:
    try:
        df.loc[i]
    except:
        df.loc[i] = [np.nan,np.nan,np.nan,np.nan]
df = df.sort_values(by='Time')

fig, axs = plt.subplots(2,1,sharex=True)
df[['HR','SBP','DBP']].plot(style='o', ax=axs[0])
df[['PTT']].plot(style='o', ax=axs[1])

df[['HR','SBP','DBP']] = df[['HR','SBP','DBP']].interpolate(method='polynomial',order=1)
df['PTT'] = df['PTT'].interpolate(method='polynomial',order=1)

[axs[0].plot(df[x],'*',alpha=.4,label=y) for x,y in zip(['HR','SBP','DBP'],['HR - resampled','SBP - resampled','DBP - resampled'])]
axs[1].plot(df['PTT'],'*',alpha=.4,label='PTT - resampled')
[axs[i].legend() for i in range(2)]
axs[1].set_xlabel('Time [s]')
plt.tight_layout()
# plt.show()
plt.close()

df = df.loc[x_final].dropna()

TRAIN_PERC = 0.85
train_size = int(TRAIN_PERC*len(df.index))

# separate the input features (HR, PPG) from the target variables (DBP, SBP)
X = df[['HR', 'PTT']]
Y = df[['DBP', 'SBP']]

# Normaliziation
x = X.values
y = Y.values
print(f'shape of y {y.shape}')
mmx = x.mean(axis=0)
ssx = x.std(axis=0)
X_scaled = (x-mmx)/ssx

mmy = y.mean(axis=0)
ssy = y.std(axis=0)
y_scaled = (y-mmy)/ssy

# Divide the data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
# print(f"shape of y train:{y_train.shape}")

# Grid search
param_grid = {
  'u_conv1_1':[64, 128],
  'u_conv2_1':[64, 128],
  'u_LSTM':[64, 128]
}
run_gridsearch(X_train,y_train, param_grid)

# Train the model
# model = training(X_train,X_val,y_train,y_val)

# y_hat = model.predict(X_scaled)

# y_hat = y_hat * ssy + mmy
# y = y_scaled * ssy + mmy

# plot_all(y, y_hat)


