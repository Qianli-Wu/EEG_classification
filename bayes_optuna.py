import optuna
import argparse
import keras
import csv
from data_preprocess import load_data
from keras_model import hybrid_cnn_lstm_model, create_cnn_transformer_model, transformer_model


def keras_train(args, csv=False):

    print(f'learning_rate: {args.learning_rate}')
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data(onehot=True)


    # Model parameters
    learning_rate = args.learning_rate
    epochs = args.epoch
    num_heads = args.num_heads
    hybrid_cnn_lstm_optimizer = keras.optimizers.Adam(lr=learning_rate)

    model = create_cnn_transformer_model(num_heads=num_heads)
    # model = transformer_model()
    # model = hybrid_cnn_lstm_model

    model.summary()

    # Compiling the model
    model.compile(loss='categorical_crossentropy',
                    optimizer=hybrid_cnn_lstm_optimizer,
                    metrics=['accuracy'])

    # Training and validating the model
    results = model.fit(x_train,
                y_train,
                batch_size=64,
                epochs=epochs,
                validation_data=(x_valid, y_valid), verbose=True)
    if csv: 
        data = [["EEG", str(args.runs), str(args.epoch), str(args.learning_rate), str(results.history['val_accuracy'][-1])]]

        with open('testData.csv', "a") as file:
            writer = csv.writer(file)
            writer.writerows(data)
            print("write succeed")

    cnn_score = model.evaluate(x_test, y_test, verbose=0)  # TODO: delete this line before submitting
    print('Test accuracy of the CNN + Transformer model:',cnn_score[1])

    return results.history['val_accuracy'][-1]

def keras_cnn_transformer(args, csv=False):

    print(f'learning_rate: {args.learning_rate}')
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data(onehot=True)


    # Model parameters
    learning_rate = args.learning_rate
    epochs = args.epoch
    num_heads = args.num_heads
    hybrid_cnn_transformer_optimizer = keras.optimizers.Adam(lr=learning_rate)

    hybird_cnn_transformer_model = create_cnn_transformer_model(num_heads=num_heads)
    # Print the model summary
    hybird_cnn_transformer_model.summary()


    # Compiling the model
    hybird_cnn_transformer_model.compile(loss='categorical_crossentropy',
                    optimizer=hybrid_cnn_transformer_optimizer,
                    metrics=['accuracy'])

    # Training and validating the model
    hybrid_cnn_transformer_model_results = hybird_cnn_transformer_model.fit(x_train,
                y_train,
                batch_size=64,
                epochs=epochs,
                validation_data=(x_valid, y_valid), verbose=True)

    if csv:
        data = [["EEG", str(args.runs), str(args.epoch), str(args.learning_rate), str(hybrid_cnn_transformer_model_results.history['val_accuracy'][-1])]]

        with open('testData.csv', "a") as file:
            writer = csv.writer(file)
            writer.writerows(data)
            print("write succeed")

    cnn_score = hybird_cnn_transformer_model.evaluate(x_test, y_test, verbose=0)  # TODO: delete this line before submitting
    print('Test accuracy of the CNN + Transformer model:',cnn_score[1])

    return hybrid_cnn_transformer_model_results.history['val_accuracy'][-1]

def optuna_objective(trail):

    epochs = trail.suggest_int("epochs", 45, 55, 5)
    learning_rate = trail.suggest_float("learning_rates", 9e-4, 1e-3)

    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', dest='runs', type=int, default=1)
    parser.add_argument('--epoch', dest='epoch', type=int, default=epochs)
    parser.add_argument('--learning_rate', type=float, default=learning_rate)

    args = parser.parse_args()

    val_accuracy = keras_train(args)
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



if __name__ == "__main__":

    # open file to store the data
    header = ["Dataset", "Run", "Epoch", "Learning_Rate"]
    with open("testData.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(header)

    # Use optuna to tune hyperparameters
    best_params, best_score = optimizer_optuna(10, "TPE")
    print(best_params, best_score)