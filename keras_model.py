import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Dropout
from keras.layers import Conv1D, Conv2D,LSTM, BatchNormalization,MaxPooling2D,Reshape
from keras.layers import MultiHeadAttention, Permute, LayerNormalization, GlobalAveragePooling1D
from keras.layers import SimpleRNN, GRU


def cnn_transformer_model(num_cnn_layer=3, filters=25, input_shape=(250, 1, 22), num_classes=4, 
                          num_transformer_layer=1, ff_dim=800, dropout=0.5, num_heads=2):
    '''
    CNN + Transfoermer Encoder Model
    '''
    inputs = keras.Input(shape=input_shape)

    # CNN layer 1
    x = cnn_layer(inputs, input_shape=input_shape, filters=filters, dropout=dropout)

    # CNN layer 2 to num_cnn_layer
    for _ in range(num_cnn_layer - 1):
        filters *= 2
        x = cnn_layer(x, filters=filters, dropout=dropout)

    # Prepare the data for the Transformer
    x = Reshape((-1, x.shape[-1]))(x)
    x = Permute((2, 1))(x)

    for _ in range(num_transformer_layer):
        x = transformer_encoder(x, x.shape[-1], num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
    
    # FC output
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)



def cnn_rnn_model(num_cnn_layer=3, filters=25, input_shape=(250, 1, 22), num_classes=4, dropout=0.5):
    '''
    CNN + GRU + SimpleRNN
    '''
    inputs = keras.Input(shape=input_shape)
    # CNN layer 1
    x = cnn_layer(inputs, input_shape=input_shape, filters=filters)

    # CNN layer 2 to num_cnn_layer
    for _ in range(num_cnn_layer - 1):
        filters *= 2
        x = cnn_layer(x, filters=filters, dropout=dropout)

    # FC layer
    x = Flatten()(x)
    x = Dense((100))(x)
    x = Reshape((100, 1))(x)

    # GRU + SimpleRNN
    x = GRU(256, return_sequences=True)(x)
    x = SimpleRNN(128)(x)

    # Output FC layer with softmax activation
    outputs = Dense(4, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)



def cnn_model(num_cnn_layer=3, filters=25, input_shape=(250, 1, 22), num_classes=4, dropout=0.5):
    '''
    Simple CNN Model
    '''
    inputs = keras.Input(shape=input_shape)
    # CNN layer 1
    x = cnn_layer(inputs, input_shape=input_shape, filters=filters)

    # CNN layer 2 to num_cnn_layer
    for _ in range(num_cnn_layer - 1):
        filters *= 2
        x = cnn_layer(x, filters=filters, dropout=dropout)
    
    # FC output
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)


def transformer_model(input_shape=(250, 1, 22), head_size=250, num_heads=2, 
                ff_dim=500, num_transformer_blocks=4, mlp_units=[128], 
                dropout=0.5, mlp_dropout=0.5,):
    '''
    Transformer Model
    '''
    
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Prepare the data for the Transformer
    x = Reshape((-1, x.shape[-1]))(x)
    x = Permute((2, 1))(x)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)

    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(4, activation="softmax")(x)
    return keras.Model(inputs, outputs)




def cnn_layer(inputs, input_shape=None, filters=25, kernel_size=(10,1), 
              padding='same', activation='elu', pool_pool_size=(3,1), 
              pool_padding='same', dropout=0.5):
    if input_shape is None:
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation)(inputs)
    else:
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation, input_shape=input_shape)(inputs)
    x = MaxPooling2D(pool_size=pool_pool_size, padding=pool_padding)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    return x


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



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res


# ===================
# NOT USED ANYMORE
# ===================
# Define the model
def create_cnn_transformer_model(num_heads=2, input_shape=(250, 1, 22), num_classes=4, ff_dim=500, dropout=0.5, time=500):
    inputs = keras.Input(shape=input_shape)

    # CNN layer 1
    x = Conv2D(filters=25, kernel_size=(10,1), padding='same', activation='elu', input_shape=(int(time/2),1,22))(inputs)
    x = MaxPooling2D(pool_size=(3,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # CNN layer 2
    x = Conv2D(filters=50, kernel_size=(10,1), padding='same', activation='elu')(x)
    x = MaxPooling2D(pool_size=(3,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)


    # # CNN layer 3
    # x = Conv2D(filters=100, kernel_size=(10,1), padding='same', activation='elu')(x)
    # x = MaxPooling2D(pool_size=(3,1), padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)


    # # CNN layer 4
    # x = Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu')(x)
    # x = MaxPooling2D(pool_size=(3,1), padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)


    # Prepare the data for the Transformer
    x = Reshape((-1, x.shape[-1]))(x)
    x = Permute((2, 1))(x)

    # Transformer layer 1
    d_model = x.shape[-1]
    x = transformer_encoder(x, d_model, num_heads, ff_dim, dropout)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)




def create_CNN_LSTM_model():
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
    
    hybrid_cnn_lstm_model.add(GRU(256, return_sequences=True))

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    hybrid_cnn_lstm_model.add(SimpleRNN(128))

    hybrid_cnn_lstm_model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation
    '''
    hybrid_cnn_lstm_model.add(LSTM(10, dropout=0.6, recurrent_dropout=0.1, input_shape=(100,1), return_sequences=False))


    # Output layer with Softmax activation 
    hybrid_cnn_lstm_model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation
    '''
    # Printing the model summary
    # hybrid_cnn_lstm_model.summary()
    return hybrid_cnn_lstm_model

