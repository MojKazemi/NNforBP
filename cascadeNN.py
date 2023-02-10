import pandas as pd
import numpy as np
import os
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

class cascadeNN():
  def __init__(self, patient):
    self.TRAIN_PERC = 0.95
    if not os.path.exists('./Dataset/NN_Cascade_model'):
      os.mkdir('./Dataset/NN_Cascade_model')
    self.regr_path = './Dataset/NN_Cascade_model/'

    if not os.path.exists(self.regr_path+'Plots'):
      os.mkdir(self.regr_path+'Plots')
    if not os.path.exists(self.regr_path + 'Final_Model'):
        os.mkdir(self.regr_path + 'Final_Model/')

    self.plot_path = self.regr_path + 'Plots/'
    self.final_model = self.regr_path + 'Final_Model/'
    self.patient = patient
    df = pd.read_csv(f'./Dataset/Regression/{self.patient}.csv').set_index('Time')
    self.df = df.dropna(how='all')

  def data_interpolate(self,df):
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
    self.time = df.index.values
    return df

  def data_prepare(self):

    data={'Train':{},'Val':{},'Test':{},'Sep':{},'Dataset':{}}
    df = self.data_interpolate(self.df)
    train_size = int(self.TRAIN_PERC*len(self.df.index))

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

    self.mmy = y.mean(axis=0)
    self.ssy = y.std(axis=0)
    y_scaled = (y-self.mmy)/self.ssy

    # Divide the data into train, validation, and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.10, random_state=42)
    X_train, data['Test']['x'], y_train, data['Test']['y'] = train_test_split(X_scaled, y_scaled, test_size=0.10, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    data['Train']['x'],data['Val']['x'],data['Train']['y'],data['Val']['y'] = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    # print(f"shape of y train:{y_train.shape}")
    data['Dataset']['x'], data['Dataset']['y'] = X_scaled, y_scaled
    return data 

  def get_model(self, u_conv1_1=128,u_conv2_1=64,u_LSTM = 64):

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

  def plot_all(self, data, model):
    y_hat = model.predict(data['Dataset']['x'])

    y_hat = y_hat * self.ssy + self.mmy
    y = data['Dataset']['y'] * self.ssy + self.mmy

    fig, axs = plt.subplots(2, 3)
    for i in range(2):
      if i==0:
          _title = 'DBP'
      else:
          _title = 'SBP'

      # plot regression plot 
      sns.regplot(ax=axs[i,0],x=y[:,i],y=y_hat[:,i], scatter_kws={'s': 2})
      axs[i,0].set_xlabel("True Values")
      axs[i,0].set_ylabel("Predicted Values")
      axs[i,0].set_title(f"Regression Plot of True Values vs. Predicted Values({_title})")

      # Plot error histogram
      error = y [:,i]- y_hat[:,i]

      axs[i,1].hist(error, bins=20, rwidth=0.8)
      axs[i,1].set_xlabel("Error(mmHg)")
      axs[i,1].set_ylabel("Frequency")
      axs[i,1].set_title(f"Histogram of Error ({_title})")

      # Plot Y and Y_hat
      axs[i,2].plot(self.time, y_hat[:,i], c='r', label = 'Y_predict')
      axs[i,2].plot(self.time, y[:,i], c='b', label = 'Y')
      axs[i,2].set_ylabel('DBP[mmHg]')
      axs[i,2].set_xlabel('Time[s]')
      axs[i,2].set_title(f'Real value of {_title} and Predict value of {_title}')
      axs[i,2].legend()

      # Plot vertical lines to represent the test region
      # axs[i,2].axvline(x=len(y_scaled), color='gray', linestyle='--')
      # axs[i,2].axvline(x=len(y), color='gray', linestyle='--')

      # Fill the space between the vertical lines
      axs[i,2].fill_betweenx([min(y[:,i]),max(y[:,i])], self.time[int(self.TRAIN_PERC*len(self.time))], self.time[-1], color='gray', alpha=0.5)            
      axs[i,2].grid()
    plt.savefig(self.plot_path+self.patient+'_Pred.png')
    plt.tight_layout()
    plt.show()

  def training(self, data):
    # num_features = 2

    model = self.get_model()

    checkpoint = ModelCheckpoint("./best_model.h5", monitor='val_loss', save_best_only=True, mode='min')

    history = model.fit(data['Train']['x'],
                        data['Train']['y'], #batch_size = 32,
                        validation_data=(data['Val']['x'], data['Val']['y']),
                        epochs=50)

    trainPredict = model.predict(data['Train']['x'])
    validPredict = model.predict(data['Val']['x'])
    testPredict = model.predict(data['Test']['x'])
    print(f'shape of train predict{trainPredict.shape}')

    trainScore = math.sqrt(mean_squared_error(data['Train']['y'][:,0], trainPredict[:,0]))
    testScore = math.sqrt(mean_squared_error(data['Test']['y'][:,0], testPredict[:,0]))

    print( 'Train Score DBP: %.2f RMSE' % (trainScore))
    print( 'Test Score DBP: %.2f RMSE' % (testScore))

    trainScore = math.sqrt(mean_squared_error(data['Train']['y'][:,1], trainPredict[:,1]))
    testScore = math.sqrt(mean_squared_error(data['Test']['x'][:,1], testPredict[:,1]))
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

  def run_gridsearch(self, data, param_grid):

    model = KerasRegressor(build_fn=self.get_model, epochs=70, batch_size=32, verbose=0)

    kfold = KFold(n_splits=5, shuffle=True)

    grid = GridSearchCV(estimator=model, param_grid=param_grid,# n_jobs=-1,
                        cv=kfold, verbose=1,
                        return_train_score=True)
    
    grid_result = grid.fit(data['Train']['x'],data['Train']['x'])
    print(f"best_estimator_: {grid_result.best_estimator_}")
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

if __name__=='__main__':
  patient = '3402408'

  CC_NN = cascadeNN(patient)
  data = CC_NN.data_prepare()
  # Grid search
  param_grid = {
    'u_conv1_1':[128],
    'u_conv2_1':[128],
    'u_LSTM':[128]
  }
  CC_NN.run_gridsearch(data, param_grid)

  # Train the model
  # model = CC_NN.training(data)

  # CC_NN.plot_all(data, model)


'''
  param_grid = {
  'u_conv1_1':[64, 128],
  'u_conv2_1':[64, 128],
  'u_LSTM':[64, 128]
}
  Best: -0.468258 using {'u_LSTM': 64, 'u_conv1_1': 128, 'u_conv2_1': 64}
'''
