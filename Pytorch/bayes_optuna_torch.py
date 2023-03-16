import optuna
import argparse
import csv
from data_preprocess import load_data, EEGDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch_model import torch_hybrid_cnn_lstm_model

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    size = len(dataloader.dataset)
    correct = 0
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

    correct /= size
    return running_loss / len(dataloader), correct*100

def torch_cnn_lstm(args):

    print(f'learning_rate: {args.learning_rate}\n')
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data()

    train_dataset = EEGDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    valid_dataset = EEGDataset(torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Model parameters
    learning_rate = args.learning_rate
    epochs = args.epoch
    # Initialize the model, loss, and optimizer
    d_model = 22
    nhead = 2
    num_layers = 2
    num_classes = 4
    model = torch_hybrid_cnn_lstm_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, valid_dataloader, criterion, device)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Acc: {val_acc:.4f}')


    data = [["EEG", str(args.runs), str(args.epoch), str(args.learning_rate), str(val_acc)]]

    with open('testData.csv', "a") as file:
        writer = csv.writer(file)
        writer.writerows(data)
        print("write succeed")

    return val_loss



def optuna_objective(trail):

    epochs = trail.suggest_int("epochs", 40, 100, 5)
    learning_rate = trail.suggest_float("learning_rates", 1e-6, 0.1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', dest='runs', type=int, default=1)
    parser.add_argument('--epoch', dest='epoch', type=int, default=epochs)
    parser.add_argument('--learning_rate', type=float, default=learning_rate)

    args = parser.parse_args()

    val_accuracy = torch_cnn_lstm(args)
    return val_accuracy





def optimizer_optuna(n_trials, algo):
    # 定义使用TPE或者GP
    if algo == "TPE":
        algo = optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24)
    # elif algo == "GP":
    #     from optuna.integration import SkoptSampler
    #     import skopt
    #     algo = SkoptSampler(skopt_kwargs={'base_estimator': 'GP',  # 选择高斯过程
    #                                       'n_initial_points': 10,  # 初始观测点10个
    #                                       'acq_func': 'EI'})  # 选择的采集函数为EI，期望增量

    # 实际优化过程，首先实例化优化器
    study = optuna.create_study(sampler=algo  # 要使用的具体算法
                                , direction="minimize"  # 优化的方向，可以填写minimize或maximize
                                )
    # 开始优化，n_trials为允许的最大迭代次数
    # 由于参数空间已经在目标函数中定义好，因此不需要输入参数空间
    study.optimize(optuna_objective  # 目标函数
                   , n_trials=n_trials  # 最大迭代次数（包括最初的观测值的）
                   , show_progress_bar=True  # 要不要展示进度条呀？
                   )

    # 可直接从优化好的对象study中调用优化的结果
    # 打印最佳参数与最佳损失值
    print("\n", "\n", "best params: ", study.best_trial.params,
          "\n", "\n", "best score: ", study.best_trial.values,
          "\n")

    return study.best_trial.params, study.best_trial.values

