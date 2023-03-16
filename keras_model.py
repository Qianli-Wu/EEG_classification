import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Dropout
from keras.layers import Conv2D,LSTM,BatchNormalization,MaxPooling2D,Reshape

# Building the CNN model using sequential class
hybrid_cnn_lstm_model = Sequential()

# Conv. block 1
hybrid_cnn_lstm_model.add(Conv2D(filters=25, kernel_size=(10,1), padding='same', activation='elu', input_shape=(250,1,22)))
hybrid_cnn_lstm_model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
hybrid_cnn_lstm_model.add(BatchNormalization())
hybrid_cnn_lstm_model.add(Dropout(0.5))

# Conv. block 2
hybrid_cnn_lstm_model.add(Conv2D(filters=50, kernel_size=(10,1), padding='same', activation='elu'))
hybrid_cnn_lstm_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
hybrid_cnn_lstm_model.add(BatchNormalization())
hybrid_cnn_lstm_model.add(Dropout(0.5))

# Conv. block 3
hybrid_cnn_lstm_model.add(Conv2D(filters=100, kernel_size=(10,1), padding='same', activation='elu'))
hybrid_cnn_lstm_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
hybrid_cnn_lstm_model.add(BatchNormalization())
hybrid_cnn_lstm_model.add(Dropout(0.5))

# Conv. block 4
hybrid_cnn_lstm_model.add(Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu'))
hybrid_cnn_lstm_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
hybrid_cnn_lstm_model.add(BatchNormalization())
hybrid_cnn_lstm_model.add(Dropout(0.5))

# FC+LSTM layers
hybrid_cnn_lstm_model.add(Flatten()) # Adding a flattening operation to the output of CNN block
hybrid_cnn_lstm_model.add(Dense((100))) # FC layer with 100 units
hybrid_cnn_lstm_model.add(Reshape((100,1))) # Reshape my output of FC layer so that it's compatible
hybrid_cnn_lstm_model.add(LSTM(10, dropout=0.6, recurrent_dropout=0.1, input_shape=(100,1), return_sequences=False))


# Output layer with Softmax activation 
hybrid_cnn_lstm_model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation


# Printing the model summary
hybrid_cnn_lstm_model.summary()




# Building the CNN model using sequential class
cnn_model = Sequential()

# Conv. block 1
cnn_model.add(Conv2D(filters=25, kernel_size=(10,1), padding='same', activation='elu', input_shape=(250,1,22)))
cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.5))

# Conv. block 2
cnn_model.add(Conv2D(filters=50, kernel_size=(10,1), padding='same', activation='elu'))
cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.5))

# Conv. block 3
cnn_model.add(Conv2D(filters=100, kernel_size=(10,1), padding='same', activation='elu'))
cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.5))

# Conv. block 4
cnn_model.add(Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu'))
cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.5))

# Output layer with Softmax activation 
cnn_model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation


# Printing the model summary
cnn_model.summary()



# Building the CNN model using sequential class
hybird_cnn_transformer_model = Sequential()

# Conv. block 1
hybird_cnn_transformer_model.add(Conv2D(filters=25, kernel_size=(10,1), padding='same', activation='elu', input_shape=(250,1,22)))
hybird_cnn_transformer_model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
hybird_cnn_transformer_model.add(BatchNormalization())
hybird_cnn_transformer_model.add(Dropout(0.5))

# Conv. block 2
hybird_cnn_transformer_model.add(Conv2D(filters=50, kernel_size=(10,1), padding='same', activation='elu'))
hybird_cnn_transformer_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
hybird_cnn_transformer_model.add(BatchNormalization())
hybird_cnn_transformer_model.add(Dropout(0.5))

# Conv. block 3
hybird_cnn_transformer_model.add(Conv2D(filters=100, kernel_size=(10,1), padding='same', activation='elu'))
hybird_cnn_transformer_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
hybird_cnn_transformer_model.add(BatchNormalization())
hybird_cnn_transformer_model.add(Dropout(0.5))

# Conv. block 4
hybird_cnn_transformer_model.add(Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu'))
hybird_cnn_transformer_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
hybird_cnn_transformer_model.add(BatchNormalization())
hybird_cnn_transformer_model.add(Dropout(0.5))

# Output layer with Softmax activation 
hybird_cnn_transformer_model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation


# Printing the model summary
hybird_cnn_transformer_model.summary()

