import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import shutil
import time
import os
import math


############### SECTION : LOAD DATA ###############
def load_training_data(predictor_col):
    
    #df_train = pd.read_csv("input/train.csv")

    df_train = pd.concat([pd.read_csv(f'Data/train/{f}') for f in os.listdir('Data/train') if f.endswith('.csv')])

    print(df_train.head(5))

    df_train.reset_index(drop=True, inplace=True)

    print("######## TRAIN DATA SUMMARY #####################")
    print(df_train.tail(5))
    print(df_train.describe())   
    print(df_train.dtypes)
       
    #train_x,train_y,val_x,val_y = df_train1, df_train1.pop(predictor_col),df_val,df_val.pop(predictor_col)
    df_val = pd.concat([pd.read_csv(f'Data/test/{f}') for f in os.listdir('Data/test') if f.endswith('.csv')]) 

    df_val.reset_index(drop=True, inplace=True)

    print("######## TEST DATA SUMMARY #####################")
    print(df_val.tail(5))
    print(df_val.describe())   
    print(df_val.dtypes)
        

    return df_train,df_val


############### SECTION : LOAD DATA ###############
def process_data_for_ratio(train_x,val_x):

    train_x_mod = pd.DataFrame()
    val_x_mod = pd.DataFrame()

    train_hl_ratio = train_x['High']/train_x['Low']
    train_x_mod["high_low_ratio"] = train_hl_ratio
    val_hl_ratio = val_x['High']/val_x['Low']
    val_x_mod["high_low_ratio"]=val_hl_ratio

    train_oc_ratio = train_x['Open']/train_x['Close']
    train_x_mod["open_close_ratio"]=train_oc_ratio
    val_oc_ratio = val_x['Open']/val_x['Close']
    val_x_mod["open_close_ratio"]=val_oc_ratio

    train_oh_ratio = train_x['Open']/train_x['High']
    train_x_mod["open_high_ratio"]=train_oh_ratio
    val_oh_ratio = val_x['Open']/val_x['High']
    val_x_mod["open_high_ratio"]=val_oh_ratio

    train_ol_ratio = train_x['Open']/train_x['Low']
    train_x_mod["open_low_ratio"]=train_ol_ratio
    val_ol_ratio = val_x['Open']/val_x['Low']
    val_x_mod["open_low_ratio"]=val_ol_ratio

    train_ch_ratio = train_x['Close']/train_x['High']
    train_x_mod["close_high_ratio"]=train_ch_ratio
    val_ch_ratio = val_x['Close']/val_x['High']
    val_x_mod["close_high_ratio"]=val_ch_ratio

    train_cl_ratio = train_x['Close']/train_x['Low']
    train_x_mod["close_low_ratio"]=train_cl_ratio
    val_cl_ratio = val_x['Close']/val_x['Low']
    val_x_mod["close_low_ratio"]=val_cl_ratio

    print("########Debug#########")
    print(train_x[['Open','High','Low','Close']].head(5).apply(np.std,axis=1))
    print("########Debug#########")

    train_std = train_x[['Open','High','Low','Close']].apply(np.std,axis=1)
    train_x_mod["std_dev"]=train_std.pct_change()
    val_std = val_x[['Open','High','Low','Close']].apply(np.std,axis=1)
    val_x_mod["std_dev"]=val_std.pct_change()

    train_pct_change = train_x[['Open','High','Low','Close','Shares Traded']].pct_change()
    train_x_mod=train_x_mod.join(train_pct_change)
    val_pct_change = val_x[['Open','High','Low','Close','Shares Traded']].pct_change()
    val_x_mod=val_x_mod.join(val_pct_change)

    #train_x_mod["Date"]=train_x["Date"]
    #val_x_mod["Date"]=val_x["Date"]



    print("######## TRAIN DATA SUMMARY #####################")
    print(train_x_mod.head(5))
    print(train_x_mod.describe())   
    print(train_x_mod.dtypes)


    print("######## VAL DATA SUMMARY #####################")
    print(val_x_mod.head(5))
    print(val_x_mod.describe())   
    print(val_x_mod.dtypes)   


    return train_x_mod.dropna(),val_x_mod.dropna()

######CHECK CORELATIONS TO CLOSE

###ARIMAX MODEL#####


######LSTM MODEL#######
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):

    attributes = dataset.shape[1]
    #dataset= np.reshape(dataset.to_numpy(),(len(dataset),look_back,-1))
    
    print("Printing first ten rows###################################################")
    print(type(dataset.iloc[11]))
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a.as_matrix())
        dataY.append(dataset.iloc[(i + look_back)]["Close"])

    #print(dataY[10])

    for i in dataX:
        if(i.shape[1])!=12:
            print("DDDDDDDDDDDDDDDDDDDDDDDDDDDDDd")
    dataXA = np.empty(len(dataX),dtype=object)
    dataYA = np.empty(len(dataY),dtype=object)

    dataXA[:]= dataX
    dataYA[:]=dataY 


    
    print(np.shape(dataXA))
    dataXA= np.array(dataXA)
    dataYA = np.array(dataYA)

    dataXNPA = np.empty((dataXA.shape[0],dataXA[0].shape[0],dataXA[0].shape[1]))
    dataYNPA = np.empty((dataYA.shape[0]))
    
    for i in range(len(dataXA)):
        dataXNPA[i]=dataXA[i]
        dataYNPA[i]=dataYA[i]

    print("DATAX SHAPE : ####################+ " + str((dataXNPA[0].shape)))

    print("DATAX SHAPE 2 : ####################+ " + str(type(dataXNPA)))


    return dataXNPA, dataYNPA

def build_model(layers):
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(
        #input_dim=layers[0],
        #output_dim=layers[1],
        input_shape=(None,layers[0]),
        #stateful=True,
        units=layers[1],
        #batch_input_shape=(100,layers[1],layers[0])
        return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(
        layers[2],
        return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(
        units=50,
        return_sequences=False)),
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=30))
    model.add(Activation("linear"))
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=10))
    model.add(Activation("linear"))
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model


####### TODO : Pass Batch Size and steps in command line args #########
def main(argv):
    # args = parser.parse_args(argv[1:])
    train_x, test_x = load_training_data("Close")

    #train_x = train_x.drop("Date",1)
    #test_x = test_x.drop("Date",1)


    

    train_x, test_x = process_data_for_ratio(train_x,test_x)

    


    print(train_x.head(5))

    columns = list(train_x.columns.values)

    scalar = MinMaxScaler(feature_range=(0, 1))
    train_x = pd.DataFrame(scalar.fit_transform(train_x),columns=columns)
    test_x = pd.DataFrame(scalar.fit_transform(test_x),columns=columns)





    

    


     #reshape into X=t and Y=t+1
    look_back = 10
    trainX, trainY = create_dataset(train_x, look_back)
    testX, testY = create_dataset(test_x, look_back)


    

    print(trainX.shape,trainY.shape)

    print(trainX)

    trainX= trainX.reshape((trainX.shape[0],look_back,-1))
    testX= testX.reshape((testX.shape[0],look_back,-1))
    

   

    model = build_model([trainX.shape[2], look_back, 100, 1])

    
    model.fit(
    trainX,
    trainY,
    batch_size=100,
    epochs=50,
    #validation_split=0.1,
    verbose=1)

    trainScore = model.evaluate(trainX, trainY, verbose=0,batch_size=100)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    #testScore = model.evaluate(testX, testY, verbose=0)
    #print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

    diff = []
    ratio = []
    pred = model.predict(trainX)

    print(len(pred))
    for u in range(len(pred)):
        pr = pred[u][0]
        ratio.append((trainY[u] / pr) - 1)
        diff.append(abs(trainY[u] - pr))

    print(diff)
    print(ratio)

 
    import matplotlib
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.subplot(211)
    plt.plot(pred, color='red', label='Prediction')
    #plt.plot(trainY, color='blue', label='Ground Truth')
    plt.legend(loc='upper left')
    #plt.show()

    #plt.plot(pred, color='red', label='Prediction')
    plt.subplot(212)
    plt.plot(trainY, color='blue', label='Ground Truth')
    plt.legend(loc='upper left')
    plt.show()

    return


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)