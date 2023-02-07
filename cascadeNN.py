import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.callbacks import ModelCheckpoint
import seaborn as sns

# df = pd.read_csv('./Dataset/puredataset/CascadeNN/alldataset.csv')
# counthr = df['HR'].count()
# countdbp = df['DBP'].count()
# countsbp = df['SBP'].count()
# print(f'number of hr:{counthr}, DBP:{countdbp},SBP: {countsbp}')

def get_model():
    import keras
    from keras.layers import Input, LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate
    from keras.models import Model
    print('*************************************')
    # Define the first input layer for the HR signal
    input1 = Input(shape=(1,1))

    # Define the first branch of the HR signal
    conv1_1 = Conv1D(filters=32, kernel_size=1,input_shape=(1, 1), activation='relu')(input1)
    
    print("------////-------Output size of layer:", conv1_1.shape)

    pool1_1 = MaxPooling1D(pool_size=1)(conv1_1)
    print("------////-------Output size of layer:", pool1_1.shape)

    conv1_2 = Conv1D(filters=32, kernel_size=1, activation='relu')(input1)
    pool1_2 = MaxPooling1D(pool_size=1)(conv1_2)

    concat1 = Concatenate()([pool1_1, pool1_2])

    conv1_3 = Conv1D(filters=32, kernel_size=1, activation='relu')(concat1)
    pool1_3 = MaxPooling1D(pool_size=1)(conv1_3)

    conv1_4 = Conv1D(filters=32, kernel_size=1, activation='relu')(concat1)
    pool1_4 = MaxPooling1D(pool_size=1)(conv1_4)

    concat2 = Concatenate()([pool1_3, pool1_4])

    # Define the second input layer for the PPG signal
    input2 =Input(shape=(1,1))
    # Define the second branch of the PPG signal
    conv2_1 = Conv1D(filters=32, kernel_size=1, activation='relu')(input2)
    pool2_1 = MaxPooling1D(pool_size=1)(conv2_1)

    conv2_2 = Conv1D(filters=32, kernel_size=1, activation='relu')(input2)
    pool2_2 = MaxPooling1D(pool_size=1)(conv2_2)

    concat3 = Concatenate()([pool2_1, pool2_2])

    conv2_3 = Conv1D(filters=32, kernel_size=1, activation='relu')(concat3)
    pool2_3 = MaxPooling1D(pool_size=1)(conv2_3)

    conv2_4 = Conv1D(filters=32, kernel_size=1, activation='relu')(concat3)
    pool2_4 = MaxPooling1D(pool_size=1)(conv2_4)

    concat4 = Concatenate()([pool2_3, pool2_4])

    # Concatenate the outputs of the two branches
    concat5 = Concatenate()([concat2, concat4])

    # Define the LSTM layer
    lstm = LSTM(units=128, return_sequences=True)(concat5)
    # Dense layer with 3 outputs
    outputs = Dense(2, activation='linear')(lstm)

    # Define the model
    model = keras.models.Model(inputs=[input1, input2], outputs=outputs)

    # Compile the model
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt)

    return model


df = pd.read_csv('./Dataset/Regression/3602766.csv').set_index('Time')
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
# print(df.head())
# df.to_csv('./mmmmm.csv')
# print(df.shape)

TRAIN_PERC = 0.85
train_size = int(TRAIN_PERC*len(df.index))

# separate the input features (HR, PPG) from the target variables (DBP, SBP)
X = df[['HR', 'PTT']]
Y = df[['DBP', 'SBP']]

# # Normalize the input features
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)
# # X_scaled = X.values

# # print(f"min{X_scaled[0].min}and max: {X_scaled[0].max}")
# # print(X_scaled)
# # breakpoint()
# # Normalize the target variables
# # print(type(y))
# y_scaled = scaler.fit_transform(y)
# # y_scaled =y.values
# # print(type(y_scaled))

# Normaliziation
x = X.values
y = Y.values

mmx = x.mean(axis=0)
ssx = x.std(axis=0)
X_scaled = (x-mmx)/ssx

mmy = y.mean(axis=0)
ssy = y.std(axis=0)
y_scaled = (y-mmy)/ssy


# Divide the data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


X_train_reshape = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_reshape = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_val_reshape = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

timesteps = df.shape[0]
num_features = 2

model = get_model()
# Train the model

checkpoint = ModelCheckpoint("./best_model.h5", monitor='val_loss', save_best_only=True, mode='min')

history = model.fit([X_train_reshape[:, 0,:], X_train_reshape[:, 1,:]],
                    y_train, batch_size = 32,
                    validation_data=([X_val_reshape[:, 0,:],X_val_reshape[:, 1,:]], y_val),
                    epochs=100)

# Evaluate the model on the test data
# loss, accuracy = model.evaluate([X_test_reshape[:, 0,:], X_test_reshape[:, 1,:]], y_test, verbose=0)

# # Print the loss and accuracy of the model on the test data
# print("Loss:", loss)
# print("Accuracy:", accuracy)

# make predictions
trainPredict = model.predict([X_train_reshape[:, 0,:], X_train_reshape[:, 1,:]])
validPredict = model.predict([X_val_reshape[:, 0,:], X_val_reshape[:, 1,:]])
testPredict = model.predict([X_test_reshape[:, 0,:], X_test_reshape[:, 1,:]])

trainScore = math.sqrt(mean_squared_error(y_train, trainPredict[:,0]))
testScore = math.sqrt(mean_squared_error(y_test, testPredict[:,0]))

print( 'Train Score: %.2f RMSE' % (trainScore))
print( 'Test Score: %.2f RMSE' % (testScore))

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

X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))
y_hat = model.predict([X_scaled[:, 0,:], X_scaled[:, 1,:]])

y_hat = y_hat * ssy+mmy
y = y_scaled * ssy+mmy

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




'''
    predicty = np.reshape(predicty,(predicty.shape[0],predicty.shape[2]))

    # scaler.fit(y)
    denormalized_predictions = scaler.inverse_transform(predicty)
    y_np=y.values

    print(y_scaled.shape,predicty.shape)
    print(denormalized_predictions)
    plt.scatter(y_np[:,0], denormalized_predictions[:,0])
    # Plot the linear line
    z = np.polyfit(y_np[:,0], denormalized_predictions[:,0], 1)
    p = np.poly1d(z)
    plt.plot(y_np[:,0], p(y_np[:,0]),"g--")
    plt.xlabel("Real Values")
    plt.ylabel("predicty")
    plt.show()

    plt.scatter(y_np[:,1], denormalized_predictions[:,1])
    coefficients = np.polyfit(y_np[:,1], denormalized_predictions[:,1], 1)
    plt.plot(y_np[:,1], coefficients[0] * y_np[:,1] + coefficients[1], '-r')
    plt.xlabel("Real Values")
    plt.ylabel("predicty")
    plt.show()


    plt.figure()
    plt.plot(y_np[:,0],c='r')
    plt.plot( denormalized_predictions[:,0], c='b')
    plt.show()
'''