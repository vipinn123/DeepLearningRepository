# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import isnan
import shutil

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os


# Any results you write to the current directory are saved as output.

############### SECTION : LOAD DATA ###############
def load_training_data(predictor_col):
    
    df_train = pd.read_csv("input/train.csv")
   
    print("########## THE TRAIN DATA ##########")

    print(df_train.head(5))

    print("    ")

    df_train1=df_train.sample(frac=0.8,random_state=200)
    df_val=df_train.drop(df_train1.index)

    df_train1.pop('PassengerId')
    
    train_x,train_y,val_x,val_y = df_train1, df_train1.pop(predictor_col),df_val,df_val.pop(predictor_col)

    

    return (train_x,train_y,val_x,val_y)


def load_submission_data(id_col):
    
    df_test = pd.read_csv("input/test.csv")
    
    print("########## THE TEST DATA ##########")
    print(df_test.head(5))
    print("    ")

    test_x,id_col = df_test, df_test.pop(id_col)

    return (test_x,id_col)
    

############### SECTION : EXPLORE DATA ###############

def data_explore(data, data_set_name="TRAIN"):

    print("########## SUMMARY OF " + data_set_name + " DATA ##########")
    print(data.describe())
    print("    ")
       
    # TODO print dtype of columns
    print("########## DATATYPE OF " + data_set_name + " DATA ##########")
    print(data.dtypes)
    print("    ") 
    
    return

def plot_hist(data,field,title,xlabel,ylabel,bins=5):
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(data[field],bins = bins)
    #Labels and Title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return

def plot_scatter(data,title,xlabel,ylabel):
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(data[xlabel],data[ylabel])
    #Labels and Title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return


####### TODO : Check how to pass a data frame to get multiple plots on a single graph ###########
def plot_line(data,title,xlabel,ylabel,color='blue'):
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(data[xlabel],data[ylabel])
    #Labels and Title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return

############### SECTION : PRE-PROCESS INPUT DATA ###############


def process_data_for_cat(train_x,val_x,test_x,CATEGORICAL_TO_NUM_COL_LIST):

    for col in CATEGORICAL_TO_NUM_COL_LIST:
        train_x[col] = train_x[col].astype('category')
        train_x[col+"_cat"]=train_x[col].cat.codes        
        train_x.pop(col)

        val_x[col] = val_x[col].astype('category')
        val_x[col+"_cat"]=val_x[col].cat.codes        
        val_x.pop(col)

        test_x[col] = test_x[col].astype('category')
        test_x[col+"_cat"]=test_x[col].cat.codes
        test_x.pop(col)
        
    return (train_x, val_x,test_x)
   

def process_data_fill_nan_with_string(train_x,val_x,test_x,TO_FILL_COL_LIST,value='NaN'):

    for col in TO_FILL_COL_LIST:
        train_x[col] = train_x[col].fillna(value)
        val_x[col] = val_x[col].fillna(value)
        test_x[col] = test_x[col].fillna(value)
        
    return (train_x, val_x,test_x)

def process_data_fill_nan_with_mean(train_x,val_x,test_x,TO_FILL_COL_LIST,value='mean'):

    for col in TO_FILL_COL_LIST:
        train_x[col] = train_x[col].fillna(train_x[col].mean())
        val_x[col] = val_x[col].fillna(val_x[col].mean())
        test_x[col] = test_x[col].fillna(test_x[col].mean())
        
    return (train_x, val_x, test_x)

def process_data_fill_nan_with_num(train_x,val_x,test_x,TO_FILL_COL_LIST,value=0):

    for col in TO_FILL_COL_LIST:
        train_x[col] = train_x[col].fillna(value)
        val_x[col] = val_x[col].fillna(value)
        train_x[col] = train_x[col].fillna(value)
        
    return (train_x, val_x,test_x)
####### TODO : ######## 


def process_data_for_one_hot_enc(train_x,val_x,test_x,ONE_HOT_COL_LIST):

    for col in ONE_HOT_COL_LIST:
        #get a list of unique from both train and test combined
        unique_val = train_x[col].append(test_x[col]).unique()
           
        train_x[col]=train_x[col].astype('category',categories=unique_val)
        one_hot = pd.get_dummies(train_x[col],prefix=col)
        train_x.drop(col,axis=1,inplace=True)
        train_x = train_x.join(one_hot)
        
        val_x[col]=val_x[col].astype('category',categories=unique_val)
        one_hot = pd.get_dummies(val_x[col],prefix=col)
        val_x.drop(col,axis=1,inplace=True)
        val_x = val_x.join(one_hot)

        test_x[col]=test_x[col].astype('category',categories=unique_val)
        one_hot = pd.get_dummies(test_x[col],prefix=col)
        test_x.drop(col,axis=1,inplace=True)
        test_x= test_x.join(one_hot)    

    return (train_x, val_x, test_x)



####### TODO : GET TRAINING DATA BATCH ######## 


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

def get_normalization_parameters(train_x,TO_NORMALIZE_COL_LIST):
    normalization_parameters={}

    def _z_score_params(column):
        mean = train_x[column].mean()
        std = train_x[column].std()
        return {'mean': mean, 'std': std}

    for column in TO_NORMALIZE_COL_LIST:
        normalization_parameters[column] = _z_score_params(column)
  
    return normalization_parameters

def process_data_for_normalization(train_x,val_x,test_x,TO_NORMALIZE_COL_LIST,normalization_parameters):

    for column_name in TO_NORMALIZE_COL_LIST:
        column_params = normalization_parameters[column_name]
        mean = column_params['mean']
        std = column_params['std']

        train_x[column_name+'norm']=(train_x[column_name]-mean)/std
        val_x[column_name+'norm']=(val_x[column_name]-mean)/std
        test_x[column_name+'norm']=(test_x[column_name]-mean)/std

        train_x.drop(column_name,axis=1,inplace=True)
        val_x.drop(column_name,axis=1,inplace=True)
        test_x.drop(column_name,axis=1,inplace=True)

    return (train_x,val_x,test_x)

############### SECTION : PLOT MODEL PERFORMANCE RESULTS ###############

def result_plots(results,title):
    plt.plot(results.results['acc'])
    plt.plot(results.results['val_acc'])
    plt.title('Accuracy ' + title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.plot(results.results['loss'])
    plt.plot(results.results['val_loss'])
    plt.title('Loss ' + title)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    return
    
############### SECTION : MODEL BUILD, EVALUATE AND SCORE ###############

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.

    
    
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

####### TODO : Pass Batch Size and steps in command line args #########
def main(argv):
    # args = parser.parse_args(argv[1:])

    DROP_COL_LIST = ['Name','Ticket','Cabin']
    CATEGORICAL_TO_NUM_COL_LIST =['Sex']
    TO_FILL_NAN_COL_LIST =['Sex','Embarked']
    TO_FILL_MEAN_COL_LIST = ['Age','Fare']
    ONE_HOT_COL_LIST =['Embarked']
    TO_NORMALIZE_COL_LIST=['Age','Fare']
    model_dir='./summaries/train'
    

    #Load data and labes
    (train_x,train_y,val_x,val_y)=load_training_data(predictor_col="Survived")
    ##### TODO : Split training data into train and eval #######

    (test_x,id_col)=load_submission_data(id_col="PassengerId")
    
    #explore data summary and data types
    data_explore(train_x,data_set_name="TRAIN")
    #Ampute missing values
    (train_x,val_x,test_x)=process_data_fill_nan_with_string(train_x, val_x, test_x, TO_FILL_NAN_COL_LIST)
    (train_x,val_x,test_x)=process_data_fill_nan_with_mean(train_x,val_x,test_x,TO_FILL_MEAN_COL_LIST)

    (train_x,val_x,test_x)=process_data_for_cat(train_x,val_x,test_x,CATEGORICAL_TO_NUM_COL_LIST)
    train_y = train_y.astype('category').cat.codes

    #Convert to One Hot Encoding
    (train_x,val_x,test_x)=process_data_for_one_hot_enc(train_x,val_x,test_x,ONE_HOT_COL_LIST)
    
    nomalization_parameters=get_normalization_parameters(train_x,TO_NORMALIZE_COL_LIST)
    (train_x,val_x,test_x)=process_data_for_normalization(train_x,val_x,test_x,TO_NORMALIZE_COL_LIST,nomalization_parameters)
    print(train_x.head(5))
    print(test_x.head(5))
    #Drop description columns like Name, Comments, Remarks, etc
    train_x.drop(DROP_COL_LIST,axis=1,inplace=True)
    val_x.drop(DROP_COL_LIST,axis=1,inplace=True)
    test_x.drop(DROP_COL_LIST,axis=1,inplace=True)
 
    print(train_x.dtypes)
    
    model_feature_columns = []
    
    for key in train_x.keys():
        model_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build the NN Classifier.
    ####### TODO : Try out various optimizers , activation functions
    classifier = tf.estimator.DNNClassifier(
        feature_columns=model_feature_columns,
        hidden_units=[10, 10,10],
        model_dir='./summaries/train',
        n_classes=2,
        config=tf.estimator.RunConfig().replace(save_summary_steps=10))
    
    shutil.rmtree(model_dir, ignore_errors = True)
    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, batch_size=64),
        steps=500
        )
    
    """eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(train_x, train_y, 64))
    print('\nTest set accuracy: {accuracy:0.4f}\n'.format(**eval_result))
    print('train complete')"""

    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(val_x, val_y, 64))
    print('\nTest set accuracy: {accuracy:0.4f}\n'.format(**eval_result))
    print('train complete')

    predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(test_x, labels=None, batch_size=64))

    prediction_ids = [prediction['class_ids'][0] for prediction in predictions]

    submission = pd.DataFrame({
      "PassengerId": id_col,
      "Survived": prediction_ids
      })

    submission.to_csv("./nn_submission.csv", index=False)


    return
    

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)