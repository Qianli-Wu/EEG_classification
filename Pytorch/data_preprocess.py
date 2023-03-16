import numpy as np
from torch.utils.data import Dataset


# Define your dataset class
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def data_prep(X,y,sub_sample,average,noise):
    
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,500)
    X = X[:,:,0:500]
    # print('Shape of X after trimming:',X.shape)
    
    # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    
    
    total_X = X_max
    total_y = y
    # print('Shape of X after maxpooling:',total_X.shape)
    
    # Averaging + noise 
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    # print('Shape of X after averaging+noise and concatenating:',total_X.shape)
    
    # Subsampling
    
    for i in range(sub_sample):
        
        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
            
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
        
    
    # print('Shape of X after subsampling and concatenating:',total_X.shape)
    return total_X,total_y

def load_data(debug = False, onehot=False):

    ## Loading the dataset
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    person_train_valid = np.load("person_train_valid.npy")
    X_train_valid = np.load("X_train_valid.npy")
    y_train_valid = np.load("y_train_valid.npy")
    person_test = np.load("person_test.npy")

    ## Adjusting the labels so that 

    # Cue onset left - 0
    # Cue onset right - 1
    # Cue onset foot - 2
    # Cue onset tongue - 3

    y_train_valid -= 769
    y_test -= 769


    ## Random splitting and reshaping the data
    # First generating the training and validation indices using random splitting

    ind_valid = np.random.choice(2115, 375, replace=False)
    ind_train = np.array(list(set(range(2115)).difference(set(ind_valid))))

    # Creating the training and validation sets using the generated indices
    (X_train, X_valid) = X_train_valid[ind_train], X_train_valid[ind_valid] 
    (y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]


    ## Preprocessing the dataset
    x_train,y_train = data_prep(X_train,y_train,2,2,True)
    x_valid,y_valid = data_prep(X_valid,y_valid,2,2,True)
    X_test_prep,y_test_prep = data_prep(X_test,y_test,2,2,True)

    if debug:
        print('Shape of training set:',x_train.shape)
        print('Shape of validation set:',x_valid.shape)
        print('Shape of training labels:',y_train.shape)
        print('Shape of validation labels:',y_valid.shape)
        print('Shape of testing set:',X_test_prep.shape)
        print('Shape of testing labels:',y_test_prep.shape)


    if onehot:
        # Converting the labels to categorical variables for multiclass classification
        y_train = to_categorical(y_train, 4)
        y_valid = to_categorical(y_valid, 4)
        y_test = to_categorical(y_test_prep, 4)
        if debug:
            print('Shape of training labels after categorical conversion:',y_train.shape)
            print('Shape of validation labels after categorical conversion:',y_valid.shape)
            print('Shape of test labels after categorical conversion:',y_test.shape)

    # Adding width of the segment to be 1
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
    x_test = X_test_prep.reshape(X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1)
    if debug:
        print('Shape of training set after adding width info:',x_train.shape)
        print('Shape of validation set after adding width info:',x_valid.shape)
        print('Shape of test set after adding width info:',x_test.shape)

    # N, C, H, W
    if onehot:
        # Reshaping the training and validation dataset
        x_train = np.swapaxes(x_train, 1,3)
        x_train = np.swapaxes(x_train, 1,2)
        x_valid = np.swapaxes(x_valid, 1,3)
        x_valid = np.swapaxes(x_valid, 1,2)
        x_test = np.swapaxes(x_test, 1,3)
        x_test = np.swapaxes(x_test, 1,2)
        if debug:
            print('Shape of training set after dimension reshaping:',x_train.shape)
            print('Shape of validation set after dimension reshaping:',x_valid.shape)
            print('Shape of test set after dimension reshaping:',x_test.shape)

    return x_train, x_valid, x_test, y_train, y_valid, y_test


