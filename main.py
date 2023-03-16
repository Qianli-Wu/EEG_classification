import numpy as np
import pandas as pd
import argparse
import keras
import csv
import matplotlib.pyplot as plt
from data_preprocess import load_data
from keras_model import hybrid_cnn_lstm_model
from bayes_optuna import optimizer_optuna



if __name__ == "__main__":

    # open file to store the data
    header = ["Dataset", "Run", "Epoch", "Learning_Rate"]
    with open("testData.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(header)

    best_params, best_score = optimizer_optuna(10, "TPE")
    print(best_params, best_score)



