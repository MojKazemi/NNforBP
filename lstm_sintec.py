# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import tensorflow as tf
import keras
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, BatchNormalization, Dropout, Activation
from keras.layers.convolutional import MaxPooling1D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from keras.models import load_model
import joblib

class lstm_sintec(object):
    '''
     LSTM NN Model for Prediction the Blood Presure
    '''
    def __init__(self,patient):
        self.TRAIN_PERC = 0.85
        if not os.path.exists('./Dataset/NN_model'):
          os.mkdir(self.regr_path+'NN_model')
        self.regr_path = './Dataset/NN_model/'

        if not os.path.exists(self.regr_path+'Plots'):
          os.mkdir(self.regr_path+'Plots')
        if not os.path.exists(self.regr_path+'Final_Model'):
            os.mkdir(self.regr_path + 'Final_Model/')

        self.plot_path = self.regr_path+'Plots'
        self.final_model = self.regr_path + 'Final_Model/'
        self.patient = patient
        self.df = pd.read_csv(self.regr_path + patient+'.csv').dropna() 
        input_col = ['Tr', 'SPs_new', 'UpTime', 'BTB_PPG', 'PPG_h',
                     'R', 'BTB_R', 'P', 'T', 'Q', 'S', 'HR']
        output_col = ['DBP', 'SBP']
        self.X = self.df[input_col]
        self.y = self.df[output_col]

    def data_prepare(self):
        train_size = int(self.TRAIN_PERC*len(self.df.index))

        # Normaliziation
        x = self.X.values
        y = self.y.values

        self.mmx = np.around(x.mean(axis=0),2)
        self.ssx = np.around(x.std(axis=0),2)
        X_scaled = np.around((x-self.mmx)/self.ssx,3)

        self.mmy = np.around(y.mean(axis=0),2)
        self.ssy = np.around(y.std(axis=0),2)
        y_scaled = np.around((y-self.mmy)/self.ssy,3)


        print(f"Number of features before PCA: {X_scaled.shape[1]}")
        # Apply PCA to the data
        pca = PCA(n_components=0.95)
        X_scaled = pca.fit_transform(X_scaled)

        # Save the PCA model
        joblib.dump(pca, self.final_model+'pca.joblib')
        TRAIN_PERC = 0.95
        train_size = int(TRAIN_PERC*len(X_scaled))

        x_Sep, y_Sep = X_scaled[train_size:], y_scaled[train_size:]
        X_scaled, y_scaled = X_scaled[:train_size], y_scaled[:train_size]

        # Divide the data into train, and test sets
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.10, random_state=0)

        # reshape dataset
        xtrain_reshape = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_Sep_re = np.reshape(x_Sep,(x_Sep.shape[0],x_Sep.shape[1],1))
        xtest_reshape = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        return xtrain_reshape, xtest_reshape, y_train, y_test, X_scaled, y_scaled, x_Sep_re, y_Sep

    def get_model(self, n_features, units1=128,units2=128,units3=128,_learningRate=.01):
        def scaled_sigmoid(x):
          return 400 * (1 / (1 + tf.exp(-x)))

        keras.utils.get_custom_objects()['scaled_sigmoid'] = Activation(scaled_sigmoid)

        # custom_objects = {'scaled_sigmoid': Activation(scaled_sigmoid)}
        # keras.utils.get_custom_objects().update(custom_objects)

        model = Sequential(name='model_LSTM')
        model.add(Conv1D(units1, kernel_size=2,
                  activation='relu', input_shape=(n_features, 1), name='Conv1D_1'))
        model.add(MaxPooling1D(pool_size=2, strides=2, name='MaxPooling1D_1'))
        # model.add(BatchNormalization())
        model.add(Conv1D(units2, kernel_size=2,
                  activation='relu', name='Conv1D_2'))
        model.add(MaxPooling1D(pool_size=2, strides=2, name='MaxPooling1D_2'))
        model.add(LSTM(units3, activation='tanh', name='LSTM'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='linear', name='Dense'))  # scaled_sigmoid

        opt = keras.optimizers.Adam(learning_rate =_learningRate)
        model.compile(loss='mean_squared_error',
                      optimizer=opt, metrics=['accuracy'])
        return model
  
    def history_plot(self, history):
        # plot_history(history)
        plt.figure()
        print(history.history.keys())
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(self.plot_path + f'{patient}_loss.png')
        plt.show()
    
    def plot_pred(self,model,X_scaled,y_scaled, y_Sep, y_Sep_hat):

        y_hat = model.predict(X_scaled)
        y_hat = y_hat * self.ssy + self.mmy
        y = y_scaled * self.ssy + self.mmy

        y = np.concatenate((y,y_Sep), axis=0)
        y_hat = np.concatenate((y_hat,y_Sep_hat), axis=0)

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
            axs[i,2].plot(y_hat[:,i], c='r', label = 'Y_predict')
            axs[i,2].plot(y[:,i], c='b', label = 'Y')
            axs[i,2].set_ylabel('DBP[mmHg]')
            axs[i,2].set_title(f'Real value of {_title} and Predict value of {_title}')
            axs[i,2].legend()

            # Plot vertical lines to represent the test region
            axs[i,2].axvline(x=len(y_scaled), color='gray', linestyle='--')
            axs[i,2].axvline(x=len(y), color='gray', linestyle='--')

            # Fill the space between the vertical lines
            axs[i,2].fill_betweenx([min(y[:,i]),max(y[:,i])], len(y_scaled), len(y), color='gray', alpha=0.5)            
            axs[i,2].grid()
        plt.tight_layout()
        plt.show()
                
    def main(self,units1=20,units2=20,units3=35,_learningRate=.01, SelectSBP=True):
        '''
          SelectSBP = True :SBP otherwise DBP
        '''
        xtrain_reshape, _, ytrain, _,_,_,_,_ = self.data_prepare()

        _, n_features,_ = xtrain_reshape.shape

        print(f"Number of features after PCA: {xtrain_reshape.shape[1]}")

        # Make Model
        model = self.get_model(n_features, units1,units2,units3,_learningRate)
        model.summary()

        history = model.fit(xtrain_reshape, 
                            ytrain,
                            epochs=30, 
                            validation_split=0.15, 
                            verbose=0)
        model.save(self.final_model + 'model5.h5')

        self.history_plot(history)
        return history

    def run_gridsearch(self,param_grid):

        xtrain_reshape, xtest_reshape, ytrain, ytest,_,_,_,_ = self.data_prepare()
        _, n_features,_ = xtrain_reshape.shape
        print(f"Number of features after PCA: {n_features}")
        model = KerasRegressor(build_fn=self.get_model, epochs=70, batch_size=32, verbose=0)

        kfold = KFold(n_splits=5, shuffle=True)

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,
                            cv=kfold, verbose=1,
                            return_train_score=True)
        
        grid_result = grid.fit(xtrain_reshape, ytrain)
        print(f"best_estimator_: {grid_result.best_estimator_}")
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

patient = '3402408'
ls = lstm_sintec(patient=patient)
history = ls.main(units1=128, units2=128, units3=128, _learningRate=.001, SelectSBP=False)

ls = lstm_sintec(patient = patient) #
model = load_model(ls.final_model+'model5.h5')
xtrain_reshape, xtest_reshape, ytrain, ytest, X_scaled,y_scaled,x_Sep, y_Sep = ls.data_prepare()

# make predictions
trainPredict = model.predict(xtrain_reshape)
testPredict = model.predict(xtest_reshape)
y_Sep_hat = model.predict(x_Sep)

X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))

trainPredict = trainPredict * ls.ssy + ls.mmy
testPredict = testPredict * ls.ssy + ls.mmy
ytrain = ytrain * ls.ssy + ls.mmy
ytest = ytest * ls.ssy + ls.mmy
y_Sep_hat = y_Sep_hat* ls.ssy +ls.mmy
y_Sep = y_Sep*ls.ssy + ls.mmy

trainScore = math.sqrt(mean_squared_error(ytrain[:,0], trainPredict[:,0]))
print( 'Train Score [DBP]: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_Sep[:,0], y_Sep_hat[:,0]))
print( 'Test Score [DBP]: %.2f RMSE' % (testScore))

trainScore = math.sqrt(mean_squared_error(ytrain[:,1], trainPredict[:,1]))
print( 'Train Score [SBP]: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_Sep[:,1], y_Sep_hat[:,1]))
print( 'Test Score [SBP]: %.2f RMSE' % (testScore))

ls.plot_pred(model, X_scaled, y_scaled,y_Sep,y_Sep_hat)

# y_hat = model.predict(X_scaled)
# y_hat = y_hat * ls.ssy+ls.mmy
# y = y_scaled * ls.ssy+ls.mmy

# x = list(range(len(y)))
# x_Sep = [i + len(y) for i in range(len(y_Sep))]
# y = np.concatenate((y,y_Sep), axis=0)
# y_hat = np.concatenate((y_hat,y_Sep_hat), axis=0)

# for i in range(2):
#   if i==0:
#     _title = 'DBP'
#   else:
#     _title = 'SBP'

#   plt.figure()
#   plt.plot(y_hat[:,i], c='r')
#   plt.plot(y[:,i], c='b')
#   plt.show()

#   # plot regression plot 
#   plt.figure()
#   sns.regplot(x=y[:,i],y=y_hat[:,i], scatter_kws={'s': 2})
#   plt.xlabel("True Values")
#   plt.ylabel("Predicted Values")
#   plt.title(f"Regression Plot of True Values vs. Predicted Values({_title})")
#   plt.show()

#   # Plot error histogram
#   error = y [:,i]- y_hat[:,i]

#   plt.hist(error, bins=20, rwidth=0.8)
#   plt.xlabel("Error(mmHg)")
#   plt.ylabel("Frequency")

#   plt.title(f"Histogram of Error ({_title})")
#   plt.show()

# patient = '3402408'  # 'alldataset' '3400715'   '3402291'   '3402408' 3604404

# ls = lstm_sintec(patient=patient)
# param_grid = {'n_features':[8],
#               'units1': [5,10],
#               'units2': [10],
#               'units3': [4,30,128],
#               '_learningRate':[0.001, 0.0001]}
# history = ls.run_gridsearch(param_grid)


# Best: -0.971968 using {'_learningRate': 0.001, 'n_features': 8, 'units1': 128, 'units2': 32, 'units3': 32}
# Best: -0.521414 using {'_learningRate': 0.001, 'n_features': 8, 'units1': 32, 'units2': 128, 'units3': 128}