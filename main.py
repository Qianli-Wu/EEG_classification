import numpy as np
import argparse
import keras
import csv
import matplotlib.pyplot as plt
from data_preprocess import load_data
from bayes_optuna import optimizer_optuna, keras_cnn_transformer, keras_train



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
    patience = 10
    model='cnn+transformer'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=model)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=epochs)
    parser.add_argument('--learning_rate', type=float, default=learning_rate)
    parser.add_argument('--num_heads', type=int, default=num_heads)
    parser.add_argument('--patience', type=int, default=patience)

    parser.add_argument('--cnn_layers', type=int, default=3)
    parser.add_argument('--ensemble', type=int, default=3)



    args = parser.parse_args()

    acc = []
    runs = 1
    for i in range(runs):
        acc.append(keras_train(args))

    print(acc)
    print(np.mean(acc))


