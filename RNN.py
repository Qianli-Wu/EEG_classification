import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras_model import CNN_model, create_cnn_transformer_model, create_CNN_LSTM_model
from data_preprocess import load_data
from data_preprocess import data_prep

def test_RNN_model(args):
    learning_rate = 1e-3
    epochs = 50
    hybrid_cnn_lstm_optimizer = keras.optimizers.Adam(lr=learning_rate)
    x_train, x_valid, x_test, y_train, y_valid, y_test, person = load_data(onehot=True)

    hybrid_cnn_lstm_model = create_CNN_LSTM_model()
    # Compiling the model
    hybrid_cnn_lstm_model.compile(loss='categorical_crossentropy',
                    optimizer=hybrid_cnn_lstm_optimizer,
                    metrics=['accuracy'])

    # Training and validating the model
    hybrid_cnn_lstm_model_results = hybrid_cnn_lstm_model.fit(x_train,
                y_train,
                batch_size=64,
                epochs=epochs,
                validation_data=(x_valid, y_valid), verbose=True)
    hybrid_cnn_lstm_score = hybrid_cnn_lstm_model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy of the basic CNN model:',hybrid_cnn_lstm_score[1])

if __name__ == '__main__':
    test_RNN_model(None)
