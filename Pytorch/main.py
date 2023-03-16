import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt
from data_preprocess import load_data
from bayes_optuna_torch import optimizer_optuna, torch_cnn_lstm



if __name__ == "__main__":

    # open file to store the data
    header = ["Dataset", "Run", "Epoch", "Learning_Rate", "Validation Accuracy"]
    with open("testData.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(header)

    # best_params, best_score = optimizer_optuna(10, "TPE")
    # print(best_params, best_score)
    
    load_data(debug=True)
    epochs = 50
    learning_rate = 1e-3

    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', dest='runs', type=int, default=1)
    parser.add_argument('--epoch', dest='epoch', type=int, default=epochs)
    parser.add_argument('--learning_rate', type=float, default=learning_rate)

    args = parser.parse_args()
    torch_cnn_lstm(args)


