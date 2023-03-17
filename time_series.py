import numpy as np
import keras
from keras.utils import to_categorical
from keras_model import CNN_model
from data_preprocess import load_data

accuracies = []

def time_series(args):
    for time in range(200,1001,200):
        
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_data(onehot=True)
        
        basic_cnn_model = CNN_model(time)

        # Model parameters
        learning_rate = args.learning_rate
        epochs = args.epoch
        cnn_optimizer = keras.optimizers.Adam(lr=learning_rate)


        # Compiling the model
        basic_cnn_model.compile(loss='categorical_crossentropy',
                        optimizer=cnn_optimizer,
                        metrics=['accuracy'])

        # Training and validating the model
        basic_cnn_model_results = basic_cnn_model.fit(x_train,
                    y_train,
                    batch_size=64,
                    epochs=epochs,
                    validation_data=(x_valid, y_valid), verbose=True)
        
        cnn_score = basic_cnn_model.evaluate(x_test, y_test, verbose=0)
        accuracies.append(cnn_score[1])
        print('Test accuracy of the CNN model with time {0}: {1}'.format(time,cnn_score[1]))

def data_prep(X,y,sub_sample,average,noise,time):
    
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,500)
    X = X[:,:,0:time]
    
    # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)

    total_X = X_max
    total_y = y
    
    # Averaging + noise 
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    
    # Subsampling
    
    for i in range(sub_sample):
        
        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
            
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
    return total_X,total_y

def data_processing(X_train, y_train, X_valid, y_valid, X_test, y_test, time):

    ## Preprocessing the dataset
    x_train,y_train = data_prep(X_train,y_train,2,2,True,time)
    x_valid,y_valid = data_prep(X_valid,y_valid,2,2,True,time)
    X_test_prep,y_test_prep = data_prep(X_test,y_test,2,2,True,time)

    # Converting the labels to categorical variables for multiclass classification
    y_train = to_categorical(y_train, 4)
    y_valid = to_categorical(y_valid, 4)
    y_test = to_categorical(y_test_prep, 4)

    # Adding width of the segment to be 1
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
    x_test = X_test_prep.reshape(X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1)

    # Reshaping the training and validation dataset
    x_train = np.swapaxes(x_train, 1,3)
    x_train = np.swapaxes(x_train, 1,2)
    x_valid = np.swapaxes(x_valid, 1,3)
    x_valid = np.swapaxes(x_valid, 1,2)
    x_test = np.swapaxes(x_test, 1,3)
    x_test = np.swapaxes(x_test, 1,2)
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test

if __name__ == '__main__':
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    person_train_valid = np.load("person_train_valid.npy")
    X_train_valid = np.load("X_train_valid.npy")
    y_train_valid = np.load("y_train_valid.npy")
    person_test = np.load("person_test.npy")

    print ('Training/Valid data shape: {}'.format(X_train_valid.shape))
    print ('Test data shape: {}'.format(X_test.shape))
    print ('Training/Valid target shape: {}'.format(y_train_valid.shape))
    print ('Test target shape: {}'.format(y_test.shape))
    print ('Person train/valid shape: {}'.format(person_train_valid.shape))
    print ('Person test shape: {}'.format(person_test.shape))

    ## Adjusting the labels so that 

    # Cue onset left - 0
    # Cue onset right - 1
    # Cue onset foot - 2
    # Cue onset tongue - 3

    y_train_valid -= 769
    y_test -= 769


    # Model parameters
    learning_rate = 1e-3
    epochs = 50
    cnn_optimizer = keras.optimizers.Adam(lr=learning_rate)


    accuracies = []
    for time in range(100,1001,100):
        ## Random splitting and reshaping the data
        # First generating the training and validation indices using random splitting
        ind_valid = np.random.choice(2115, 375, replace=False)
        ind_train = np.array(list(set(range(2115)).difference(set(ind_valid))))

        # Creating the training and validation sets using the generated indices
        (X_train, X_valid) = X_train_valid[ind_train], X_train_valid[ind_valid] 
        (y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]
        
        X_train_set, y_train_set, X_valid_set, y_valid_set, X_test_set, y_test_set = \
                        data_processing(X_train, y_train, X_valid, y_valid, X_test, y_test, time)
        
        basic_cnn_model = CNN_model(time)
        # Compiling the model
        basic_cnn_model.compile(loss='categorical_crossentropy',
                        optimizer=cnn_optimizer,
                        metrics=['accuracy'])

        # Training and validating the model
        basic_cnn_model_results = basic_cnn_model.fit(X_train_set,
                    y_train_set,
                    batch_size=64,
                    epochs=epochs,
                    validation_data=(X_valid_set, y_valid_set), verbose=1)
        
        cnn_score = basic_cnn_model.evaluate(X_test_set, y_test_set, verbose=0)
        accuracies.append(cnn_score[1])
        print('Test accuracy of the CNN model with time {0}: {1}'.format(time,cnn_score[1]))

