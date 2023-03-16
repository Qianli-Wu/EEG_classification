import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Dropout
from keras.layers import Conv2D,LSTM,BatchNormalization,MaxPooling2D,Reshape
from keras.layers import MultiHeadAttention, Permute, LayerNormalization, GlobalAveragePooling1D

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
# hybrid_cnn_lstm_model.summary()



def CNN_model(time):
    # Building the CNN model using sequential class
    basic_cnn_model = Sequential()

    # Conv. block 1
    basic_cnn_model.add(Conv2D(filters=25, kernel_size=(10,1), padding='same', activation='elu', input_shape=(int(time/2),1,22)))
    basic_cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
    basic_cnn_model.add(BatchNormalization())
    basic_cnn_model.add(Dropout(0.5))

    # Conv. block 2
    basic_cnn_model.add(Conv2D(filters=50, kernel_size=(10,1), padding='same', activation='elu'))
    basic_cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    basic_cnn_model.add(BatchNormalization())
    basic_cnn_model.add(Dropout(0.5))

    # Conv. block 3
    basic_cnn_model.add(Conv2D(filters=100, kernel_size=(10,1), padding='same', activation='elu'))
    basic_cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    basic_cnn_model.add(BatchNormalization())
    basic_cnn_model.add(Dropout(0.5))

    # Conv. block 4
    basic_cnn_model.add(Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu'))
    basic_cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    basic_cnn_model.add(BatchNormalization())
    basic_cnn_model.add(Dropout(0.5))

    # Output layer with Softmax activation
    basic_cnn_model.add(Flatten()) # Flattens the input
    basic_cnn_model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation

    return basic_cnn_model
# # Building the CNN model using sequential class
# cnn_model = Sequential()

# # Conv. block 1
# cnn_model.add(Conv2D(filters=25, kernel_size=(10,1), padding='same', activation='elu', input_shape=(250,1,22)))
# cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
# cnn_model.add(BatchNormalization())
# cnn_model.add(Dropout(0.5))

# # Conv. block 2
# cnn_model.add(Conv2D(filters=50, kernel_size=(10,1), padding='same', activation='elu'))
# cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
# cnn_model.add(BatchNormalization())
# cnn_model.add(Dropout(0.5))

# # Conv. block 3
# cnn_model.add(Conv2D(filters=100, kernel_size=(10,1), padding='same', activation='elu'))
# cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
# cnn_model.add(BatchNormalization())
# cnn_model.add(Dropout(0.5))

# # Conv. block 4
# cnn_model.add(Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu'))
# cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
# cnn_model.add(BatchNormalization())
# cnn_model.add(Dropout(0.5))

# # Output layer with Softmax activation 
# cnn_model.add(Flatten()) # Adding a flattening operation to the output of CNN block
# cnn_model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation


# Printing the model summary
# cnn_model.summary()



# # Building the CNN model using sequential class
# hybird_cnn_transformer_model = Sequential()

# # Conv. block 1
# hybird_cnn_transformer_model.add(Conv2D(filters=25, kernel_size=(10,1), padding='same', activation='elu', input_shape=(250,1,22)))
# hybird_cnn_transformer_model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
# hybird_cnn_transformer_model.add(BatchNormalization())
# hybird_cnn_transformer_model.add(Dropout(0.5))

# # Conv. block 2
# hybird_cnn_transformer_model.add(Conv2D(filters=50, kernel_size=(10,1), padding='same', activation='elu'))
# hybird_cnn_transformer_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
# hybird_cnn_transformer_model.add(BatchNormalization())
# hybird_cnn_transformer_model.add(Dropout(0.5))

# # Conv. block 3
# hybird_cnn_transformer_model.add(Conv2D(filters=100, kernel_size=(10,1), padding='same', activation='elu'))
# hybird_cnn_transformer_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
# hybird_cnn_transformer_model.add(BatchNormalization())
# hybird_cnn_transformer_model.add(Dropout(0.5))

# # Conv. block 4
# hybird_cnn_transformer_model.add(Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu'))
# hybird_cnn_transformer_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
# hybird_cnn_transformer_model.add(BatchNormalization())
# hybird_cnn_transformer_model.add(Dropout(0.5))

# # Transformer
# hybird_cnn_transformer_model.add(Flatten()) # 800
# hybird_cnn_transformer_model.add(MultiHeadAttention(num_heads=2, key_dim=800))

# # Output layer with Softmax activation 
# hybird_cnn_transformer_model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation


# # Printing the model summary
# hybird_cnn_transformer_model.summary()


# Define the model
def create_cnn_transformer_model(num_heads=2, input_shape=(250, 1, 22), num_classes=4):
    inputs = keras.Input(shape=input_shape)

    # CNN layer 1
    x = Conv2D(filters=25, kernel_size=(10,1), padding='same', activation='elu', input_shape=(250,1,22))(inputs)
    x = MaxPooling2D(pool_size=(3,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # CNN layer 2
    x = Conv2D(filters=50, kernel_size=(10,1), padding='same', activation='elu')(x)
    x = MaxPooling2D(pool_size=(3,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)


    # CNN layer 3
    x = Conv2D(filters=100, kernel_size=(10,1), padding='same', activation='elu')(x)
    x = MaxPooling2D(pool_size=(3,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)


    # CNN layer 4
    x = Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu')(x)
    x = MaxPooling2D(pool_size=(3,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)


    # Prepare the data for the Transformer
    x = Reshape((-1, x.shape[-1]))(x)
    x = Permute((2, 1))(x)

    # Transformer layer 1
    d_model = x.shape[-1]
    x = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = LayerNormalization()(x)

    # # Transformer layer 2
    # x = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    # x = LayerNormalization()(x)

    # Classification head
    # x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)
