import numpy as np
import argparse
import keras
import csv
import matplotlib.pyplot as plt
from data_preprocess import load_data
from keras_model import hybrid_cnn_lstm_model
from time_series import time_series
from bayes_optuna import optimizer_optuna, keras_cnn_transformer, keras_train
from RNN import test_RNN_model



if __name__ == "__main__":

    # # open file to store the data
    # header = ["Dataset", "Run", "Epoch", "Learning_Rate"]
    # with open("testData.csv", "w") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(header)

    # best_params, best_score = optimizer_optuna(10, "TPE")
    # print(best_params, best_score)

    epochs = 200
    learning_rate = 4e-4
    num_heads = 2

    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', dest='runs', type=int, default=1)
    parser.add_argument('--epoch', dest='epoch', type=int, default=epochs)
    parser.add_argument('--learning_rate', type=float, default=learning_rate)
    parser.add_argument('--num_heads', type=int, default=num_heads)

    args = parser.parse_args()

    # keras_cnn_transformer(args)
# ================================================================
    #time_series(args)
    #keras_train(args)
    test_RNN_model(args)


