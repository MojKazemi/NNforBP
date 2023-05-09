# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math, os, joblib, json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import tensorflow as tf
import keras
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, BatchNormalization, Dropout, Activation, Reshape
from keras.layers.convolutional import MaxPooling1D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from keras.models import load_model



class lstm_sintec(object):
    '''
     LSTM NN Model for Prediction the Blood Presure
    '''
    def __init__(self, patient, showplot= False,root_dir =''):
        self.TRAIN_PERC = 0.75
        self.regr_path = f'{root_dir}/Dataset'
        self.check_dir(root_dir)
        self.plot_path = f'{root_dir}/Plots_ML'
        self.final_model = f'{root_dir}/Final_Model'
        self.patient = patient
        self.df = pd.read_csv(f'{self.regr_path}/{patient}.csv').dropna() 
        self.input_col = ['Tr', 'SPs_new', 'UpTime', 'BTB_PPG', 'PPG_h',
                     'R', 'BTB_R', 'P', 'T', 'Q', 'S', 'HR']
        output_col = ['DBP', 'SBP']
        self.X = self.df[self.input_col]
        self.y = self.df[output_col]
        self.time = self.df['Time'].values
        self.showplot = showplot
        plt.style.use('seaborn-darkgrid')
        self.figsize = (15,9)

    def check_dir(self,root_dir):
        if not os.path.exists(f'{root_dir}/Plots_ML'):
          os.mkdir(f'{root_dir}/Plots_ML')
        if not os.path.exists(f'{root_dir}/Final_Model'):
            os.mkdir(f'{root_dir}/Final_Model')

    def data_prepare(self):
        # Normaliziation
        x = self.X.values
        y = self.y.values
        print(x.shape)
        scale_x = StandardScaler()
        X_scaled = scale_x.fit_transform(x)

        scale_y = StandardScaler()
        y_scaled = scale_y.fit_transform(y)

        # print(f"Number of features before PCA: {X_scaled.shape[1]}")
        # Apply PCA to the data
        self.pca = PCA(n_components=0.95)
        X_scaled = self.pca.fit_transform(X_scaled)

        train_size = int(self.TRAIN_PERC*len(X_scaled))

        x_test, y_test = X_scaled[train_size:], y_scaled[train_size:]
        X_scaled, y_scaled = X_scaled[:train_size], y_scaled[:train_size]

        # Divide the data into train, and test sets
        x_train, x_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.15, random_state=0)

        data = {
            'Train':{'x':x_train,'y':y_train},
            'Val':{'x':x_val,'y':y_val},
            'test':{'x':x_test,'y':y_test},
            'Dataset':{'x':X_scaled,'y':y_scaled},
            'scaler':{'x':scale_x,'y':scale_y}
        }
        return data

    def get_model(self, n_features, units1=128,units2=128,_learningRate=.01):
        # def scaled_sigmoid(x):
        #   return 400 * (1 / (1 + tf.exp(-x)))
        # keras.utils.get_custom_objects()['scaled_sigmoid'] = Activation(scaled_sigmoid)

        model = Sequential(name='model_LSTM')
        model.add(Reshape((n_features, 1), input_shape=(n_features,)))
        model.add(Conv1D(units1, kernel_size=2,
                  activation='relu', input_shape=(n_features, 1), name='Conv1D_1'))
        model.add(MaxPooling1D(pool_size=2, strides=2, name='MaxPooling1D_1'))
        model.add(BatchNormalization())
        # model.add(Conv1D(units2, kernel_size=2,
        #           activation='relu', name='Conv1D_2'))
        # model.add(MaxPooling1D(pool_size=2, strides=2, name='MaxPooling1D_2'))
        model.add(LSTM(units2, activation='tanh', name='LSTM'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='linear', name='Dense'))  # scaled_sigmoid

        opt = keras.optimizers.Adam(learning_rate =_learningRate)
        model.compile(loss='mean_squared_error',
                      optimizer=opt, metrics=['accuracy'])
        return model
  
    def history_plot(self, history):
        # plot_history(history)
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f'{self.plot_path}/{patient}_loss.png')
        if self.showplot: plt.show()
        plt.close()
    
    def plot_pred(self,model,X_scaled,y_scaled, y_test, y_test_hat,data):

        y_hat = model.predict(X_scaled)
        y_hat = data['scaler']['y'].inverse_transform(y_hat)
        y = data['scaler']['y'].inverse_transform(y_scaled)
        # y_hat = y_hat * self.ssy + self.mmy
        # y = y_scaled * self.ssy + self.mmy

        y = np.concatenate((y,y_test), axis=0)
        y_hat = np.concatenate((y_hat,y_test_hat), axis=0)

        fig, axs = plt.subplots(2, 3)
        fig.set_size_inches(self.figsize)
        for i in range(2):
            if i==0:
                _title = 'DBP'
            else:
                _title = 'SBP'

            # plot regression plot 
            sns.regplot(ax=axs[i,0],x=y[:,i],y=y_hat[:,i], scatter_kws={'s': 2})
            axs[i,0].set_xlabel("True Values[mmHg]")
            axs[i,0].set_ylabel("Predicted Values[mmHg]")
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

            # represent the test region
            axs[i,2].fill_betweenx([min(y[:,i]),max(y[:,i])], self.time[int(self.TRAIN_PERC*len(self.time))], self.time[-1], color='gray', alpha=0.5)            
            axs[i,2].grid()
        plt.savefig(f'{self.plot_path}/{self.patient}_Pred.png')
        plt.tight_layout()
        if self.showplot: plt.show()
        plt.close()
                
    def train_model(self,units1=20,units2=35,_learningRate=.01):
        '''
          SelectSBP = True :SBP otherwise DBP
        '''
        data = self.data_prepare()

        xtrain = data['Train']['x']
        ytrain = data['Train']['y']
        _, n_features = xtrain.shape

        print(f"Number of features after PCA: {xtrain.shape[1]}")

        # Make Model
        model = self.get_model(n_features, units1,units2,_learningRate)
        tf.keras.utils.plot_model(model, to_file='conv1d.png', show_shapes=True)

        model.summary()

        print('The Model is in training mode ...')
        history = model.fit(xtrain, 
                            ytrain,
                            epochs=30, 
                            validation_data=(data['Val']['x'], data['Val']['y']), 
                            verbose=0)
        print('Training complete.')

        model.save(f'{self.final_model}/model_{self.patient}.h5')

        self.history_plot(history)   

        feature_imp = self.feature_effect(model,data)

        return history, model, feature_imp

    def check_model(self,model): 
        MAE_dict ={'DBP':{},'SBP':{}}
        data = self.data_prepare()
        # make predictions
        trainPredict = model.predict(data['Train']['x'])
        # testPredict = model.predict(data['Test']['x'])
        y_test_hat = model.predict(data['test']['x'])

        X_scaled = data['Dataset']['x'] #np.reshape(data['Dataset']['x'],(data['Dataset']['x'].shape[0],data['Dataset']['x'].shape[1],1))

        trainPredict = data['scaler']['y'].inverse_transform(trainPredict)
        ytrain = data['scaler']['y'].inverse_transform(data['Train']['y'])
        y_test_hat = data['scaler']['y'].inverse_transform(y_test_hat)
        y_test = data['scaler']['y'].inverse_transform(data['test']['y'])

        MAE_dict['DBP']['Train'] = round(mean_absolute_error(ytrain[:,0], trainPredict[:,0]))
        print( 'Train Score [DBP] for %s : %.2f MAE' % (self.patient,MAE_dict['DBP']['Train']))
        MAE_dict['DBP']['Test'] = round(mean_absolute_error(y_test[:,0], y_test_hat[:,0]))
        print( 'Test Score [DBP] for %s : %.2f MAE' % (self.patient, MAE_dict['DBP']['Test']))

        MAE_dict['SBP']['Train'] = round(mean_absolute_error(ytrain[:,1], trainPredict[:,1]))
        print( 'Train Score [SBP]: %.2f MAE' % (MAE_dict['SBP']['Train']))
        MAE_dict['SBP']['Test'] = round(mean_absolute_error(y_test[:,1], y_test_hat[:,1]))
        print( 'Test Score [SBP]: %.2f MAE' % (MAE_dict['SBP']['Test']))
        self.plot_pred(model, X_scaled, data['Dataset']['y'],y_test,y_test_hat,data)
        
        return MAE_dict

    def run_gridsearch(self,param_grid):

        data =  self.data_prepare()
        _, n_features= data['Train']['x'].shape
        param_grid['n_features'] = [n_features]

        print(f"Number of features after PCA: {n_features}")
        model = KerasRegressor(build_fn=self.get_model, epochs=70, batch_size=32, verbose=0)

        kfold = KFold(n_splits=5, shuffle=True)

        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,
                            cv=kfold, verbose=1,
                            return_train_score=True)
        
        grid_result = grid.fit(data['Train']['x'], data['Train']['y'])
        print(f"best_estimator_: {grid_result.best_estimator_}")
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        return grid_result

    def total_err(self,error_dict):
        data = error_dict
        dbp_err =[]
        sbp_err =[]
        pat = []
        for key,val in data.items():
            pat.append(key)
            dbp_err.append(val['DBP']['Test'])
            sbp_err.append(val['SBP']['Test'])

        fig,axs = plt.subplots(1,2)
        fig.set_size_inches(self.figsize)

        output={axs[0]:[dbp_err,'DBP'], axs[1]:[sbp_err, 'SBP']}
        for ax in axs:
            ax.hist(output[ax][0], bins=20, rwidth=0.8)
            ax.set_xlabel('Error(mmHg)')
            ax.set_ylabel('Frequencies')
            ax.set_title(f'Histogram of error {output[ax][1]}')
            ax.grid()
        if self.showplot: plt.show()
        plt.savefig(f'{self.plot_path}/Hist_all_MAE.png')

    def feature_effect(self,model,data):
        
        result = permutation_importance(model,data['Val']['x'], data['Val']['y'], scoring='neg_mean_squared_error')

        # Reverse PCA transformation to get feature importance scores
        importances = result.importances_mean
        inv_importances = self.pca.inverse_transform(importances)

        importances_std = result.importances_std
        inv_importances_std = self.pca.inverse_transform(importances_std)

        for index, i in enumerate(inv_importances):
            print(f"Features {self.input_col[index]} : {i:.3f} +/- {inv_importances_std[index]:.3f}")

        return (inv_importances, inv_importances_std)


if __name__=='__main__':
    patient = '3607464'#'3402408' #'3402291'#'3400715'
    ls = lstm_sintec(patient=patient, showplot= True)
    history, model = ls.train_model(units1=128, units2=128, _learningRate=.001)
    MAE_dict = ls.check_model(model)
    breakpoint()

    
    
    # if os.path.isfile('./result/MAE_result.json'):
    #     with open('./MAE_result.json','r') as f:
    #         data = json.load(f)
    #     MAE_dicts.update(data)

    # with open('./result/MAE_result.json', "w") as f:
    #     json.dump(MAE_dicts, f, indent=4)
    # print(f'Number of patients that algorithim can predict: {len(MAE_dicts)}')
    # ls.total_err(MAE_dicts)