import numpy as np
import argparse
import keras
import csv
import matplotlib.pyplot as plt
from data_preprocess import load_data
from bayes_optuna import optimizer_optuna, keras_cnn_transformer, keras_train



if __name__ == "__main__":

    learning_rate = 4e-4
    model='cnn+transformer'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=model, help='Model type to train (e.g. cnn+transformer, cnn, transformer)')
    parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning rate for the optimizer')
    parser.add_argument('--time', type=int, default=500, help='Length of the input time-series data')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads in the Transformer model')
    parser.add_argument('--patience', type=int, default=10, help='Patience for EarlyStopping during training')

    parser.add_argument('--cnn_layers', type=int, default=3, help='Number of CNN layers')
    parser.add_argument('--transformer_layers', type=int, default=1, help='Number of Transformer layers')
    parser.add_argument('--ensemble', type=int, default=3, help='Number of models to train for ensemble learning')



    args = parser.parse_args()

    keras_train(args)


