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
                ff_dim=500, num_transformer_layers=4, mlp_units=[128], 
                dropout=0.5, mlp_dropout=0.5,):
    '''
    Transformer Model
    '''
    
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Prepare the data for the Transformer
    x = Reshape((-1, x.shape[-1]))(x)
    x = Permute((2, 1))(x)

    for _ in range(num_transformer_layers):
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
    '''
    This function defines a single CNN layer for a 2D convolutional neural network. 
    The function applies a 2D convolution, followed by max pooling, batch normalization, and dropout. 
    The layer can be used as a building block for creating more complex CNN architectures.

    Parameters:
        inputs: The input tensor for the CNN layer
        input_shape (optional): The shape of the input tensor, required for the first layer of the model
        filters (optional): The number of filters in the 2D convolution
        kernel_size (optional): The size of the convolution kernel
        padding (optional): The type of padding applied to the input tensor during convolution
        activation (optional): The activation function applied after the convolution
        pool_pool_size (optional): The size of the max pooling window
        pool_padding (optional): The type of padding applied during max pooling
        dropout (optional): The dropout rate applied after batch normalization

    Returns:
        x: The output tensor after applying the CNN layer operations
    '''
    if input_shape is None:
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation)(inputs)
    else:
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation, input_shape=input_shape)(inputs)
    x = MaxPooling2D(pool_size=pool_pool_size, padding=pool_padding)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    return x



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    '''
    This function defines a Transformer encoder layer, which applies multi-head self-attention, 
    layer normalization, and a feed-forward neural network. 
    It can be used as a building block for creating complex Transformer architectures.

    Parameters:
        inputs: Input tensor
        head_size: Dimensionality of the attention head
        num_heads: Number of attention heads
        ff_dim: Hidden layer size in the feed-forward network
        dropout (optional): Dropout rate

    Returns:
        x: Output tensor after applying the Transformer encoder layer operations
    '''
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


