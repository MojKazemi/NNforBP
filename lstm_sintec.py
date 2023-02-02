# -*- coding: utf-8 -*-
"""LSTM_sintec.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13Kg5dtu08syotks5UfFvUVmx2H5KcKh_
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, LSTM, Dense
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import learning_rate_schedule
from keras.wrappers.scikit_learn import KerasRegressor


class lstm_sintec(object):
    '''
     LSTM NN Model for Prediction the Blood Presure
    '''

    def __init__(self,patient):
        self.TRAIN_PERC = 0.85
        self.regr_path = './Dataset/NN_Features/'
        self.patient = patient
        self.df = pd.read_csv(self.regr_path + patient+'.csv').dropna()


    def data_prepare(self):

        train_size = int(self.TRAIN_PERC*len(self.df.index))
        # input_col = ['Med_Tr', 'Med_SPs_new', 'Med_UpTime', 'Med_BTB_PPG', 'Med_PPG_h',
        #              'Med_R', 'Med_BTB_R', 'Med_P', 'Med_T', 'Med_Q', 'Med_S', 'HR']
        # output_col = ['Med_DBP', 'Med_SBP']
        input_col = ['Tr', 'SPs_new', 'UpTime', 'BTB_PPG', 'PPG_h',
                     'R', 'BTB_R', 'P', 'T', 'Q', 'S', 'HR']
        output_col = ['DBP', 'SBP']
        x_train, y_train = self.df[input_col].iloc[0:
                                              train_size], self.df[output_col].iloc[0:train_size]
        x_test, y_test = self.df[input_col].iloc[train_size::], self.df[output_col].iloc[train_size::]

        xtrain = x_train.values
        # ytrain_dbp = y_train['Med_DBP'].values
        # ytrain_sbp = y_train['Med_SBP'].values
        # xtest = x_test.values
        # ytest_dbp = y_test['Med_DBP'].values
        # ytest_sbp = y_test['Med_SBP'].values

        ytrain_dbp = y_train['DBP'].values
        ytrain_sbp = y_train['SBP'].values
        xtest = x_test.values
        ytest_dbp = y_test['DBP'].values
        ytest_sbp = y_test['SBP'].values

        # Normaliziation
        self.mmx = xtrain.mean(axis=0)
        self.ssx = xtrain.std(axis=0)
        xtrain = (xtrain-self.mmx)/self.ssx

        self.mmy_dbp = ytrain_dbp.mean()
        self.ssy_dbp = ytrain_dbp.std()
        ytrain_dbp = (ytrain_dbp-self.mmy_dbp)/self.ssy_dbp

        self.mmy_sbp = ytrain_sbp.mean()
        self.ssy_sbp = ytrain_sbp.std()
        ytrain_sbp = (ytrain_sbp-self.mmy_sbp)/self.ssy_sbp

        # Apply PCA to the data
        print(f"Number of features before PCA: {xtrain.shape[1]}")
        pca = PCA(n_components=0.95)
        xtrain = pca.fit_transform(xtrain)

        # reshape dataset
        xtrain_reshape = np.reshape(
            xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
        xtest_reshape = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
        return xtrain_reshape, xtest_reshape,ytrain_dbp, ytrain_sbp, ytest_dbp, ytest_sbp

    def get_model(self, n_features, units1=20, units2=20, units3=32, _learningRate=.0001):

        model = Sequential(name='model_LSTM')
        model.add(Conv1D(units1, kernel_size=2, strides=1,
                  activation='relu', input_shape=(n_features, 1), name='Conv1D_1'))
        model.add(MaxPooling1D(pool_size=2, strides=2, name='MaxPooling1D_1'))
        model.add(Conv1D(units2, kernel_size=2, strides=1,
                  activation='relu', name='Conv1D_2'))
        model.add(MaxPooling1D(pool_size=2, strides=2, name='MaxPooling1D_2'))
        model.add(LSTM(units3, activation='tanh', name='LSTM'))
        model.add(Dense(1, activation='sigmoid', name='Dense'))
        opt = keras.optimizers.Adam(learning_rate=_learningRate) # 0.0001
        model.compile(loss='mean_squared_error',
                      optimizer=opt, metrics=['accuracy'])
        return model

    def history_plot(self, history):
        # plot_history(history)
        plt.figure()
        print(history.history.keys())
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'./Plots/LSTM/{patient}_loss.png')
        plt.show()
        # plt.plot(history.history['auc'])
        # plt.plot(history.history['val_auc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()
    
    def plot_pred(self,trainPredict,testPredict):

        # shift train predictions for plotting
        df = self.df['DBP']
        df.reset_index(drop=True, inplace=True)
        trainPredictPlot = np.empty_like(df)
        trainPredictPlot[:] = np.nan
        trainPredictPlot[0:len(trainPredict)] = trainPredict[:,0]
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(df)
        testPredictPlot[:] = np.nan
        testPredictPlot[len(trainPredict):len(df)] = testPredict[:,0]
        # plot baseline and predictions
        # plt.plot(scaler.inverse_transform(dataset))
        df.plot()
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.legend(['Real Value','Train Predict', 'Test Predict'], loc='upper left')
        plt.savefig(f'./Plots/LSTM/{patient}_pred.png')
        plt.show()
        

    def main(self, units1=20, units2=20, units3=35, _learningRate=.001):
        # patient = '3400715.csv'
        xtrain_reshape, xtest_reshape,ytrain_dbp, ytrain_sbp, ytest_dbp, ytest_sbp = self.data_prepare()
        
        # Make Model
        _, n_features,_ = xtrain_reshape.shape
        print(f"Number of features after PCA: {xtrain_reshape.shape[1]}")
        model = self.get_model(n_features, units1,units2,units3,_learningRate)
        # model.summary()

        history = model.fit(xtrain_reshape, ytrain_dbp,
                            epochs=75, validation_split=0.2, verbose=0)
        # make predictions
        trainPredict = model.predict(xtrain_reshape)
        testPredict = model.predict(xtest_reshape)

        trainPredict = trainPredict * self.ssy_dbp + self.mmy_dbp
        trainY = ytrain_dbp *self.ssy_dbp + self.mmy_dbp
        testPredict = testPredict * self.ssy_dbp + self.mmy_dbp
        # testY = ytrain_dbp *self.ssy_dbp + self.mmy_dbp

        trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
        print( 'Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(ytest_dbp, testPredict[:,0]))
        print( 'Test Score: %.2f RMSE' % (testScore))
        self.history_plot(history)
        self.plot_pred(trainPredict,testPredict)

        return history

    def run_gridsearch(self):
        xtrain_reshape, xtest_reshape,ytrain_dbp, ytrain_sbp, ytest_dbp, ytest_sbp = self.data_prepare()
        _, n_features,_ = xtrain_reshape.shape

        print(f"Number of features after PCA: {n_features}")        
        model = KerasRegressor(build_fn=self.get_model, epochs=70, batch_size=32, verbose=0)

        param_grid = {'n_features':[n_features],
                      'units1': [32 ],
                      'units2': [32, 128],
                      'units3': [32, 128],
                      '_learningRate':[0.001]}
        
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
        grid_result = grid.fit(xtrain_reshape, ytrain_dbp)
        grid_result = grid.fit(xtrain_reshape, ytrain_dbp)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


if __name__ == '__main__':
    patient = '3402408'  # 'alldataset' '3400715'   '3402291'   '3402408' 3604404
    # from PreProcess_Sintec import PreProcess_Sintec
    # PrePS = PreProcess_Sintec(patient=patient)
    # PrePS.main()

    ls = lstm_sintec(patient=patient)
    # history = ls.main(units1=5, units2=5, units3=64, _learningRate=.001)
    ls.run_gridsearch()
