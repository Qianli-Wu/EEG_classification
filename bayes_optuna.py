import optuna
import argparse
import keras
import csv
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from data_preprocess import load_data
from keras_model import create_cnn_transformer_model, transformer_model, cnn_transformer_model, cnn_rnn_model, cnn_model

def ensemble_predictions(models, X_test):
    predictions = []
    for model in models:
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        predictions.append(y_pred_classes)

    # Majority vote
    predictions = np.array(predictions)
    final_predictions = stats.mode(predictions, axis=0)[0]

    return final_predictions

def keras_train(args, csv=False):

    print(f'learning_rate: {args.learning_rate}')
    x_train, x_valid, x_test, y_train, y_valid, y_test, person = load_data(onehot=True)


    # Model parameters
    model_type = args.model
    learning_rate = args.learning_rate
    epochs = args.epoch
    num_heads = args.num_heads         # Transformer
    patience = args.patience           # Early Stopping
    num_models = args.ensemble
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)


    models = []

    for i in range(num_models):
        # Sadly, it's Python3.9, which does not support Structural Pattern Matching
        if model_type == 'cnn+transformer':
            model = cnn_transformer_model(num_cnn_layer=args.cnn_layers, filters=25, num_heads=num_heads)
        elif model_type == 'cnn+rnn':
            model = cnn_rnn_model()
        elif model_type == 'transformer':
            model = transformer_model()
        elif model_type == 'cnn':
            model = cnn_model(num_cnn_layer=args.cnn_layers)

        model.summary()

        # Compiling the model
        model.compile(loss='categorical_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
        
        # Add Checkpoint
        checkpoint = ModelCheckpoint(f'model_{i}.h5', save_best_only=True)


        # Training and validating the model
        results = model.fit(x_train,
                    y_train,
                    batch_size=64,
                    epochs=epochs,
                    callbacks=[callback, checkpoint],
                    validation_data=(x_valid, y_valid), verbose=True)
        models.append(model)

    final_predictions = np.array(ensemble_predictions(models, x_test))
    # print(final_predictions)
    norml_y_test = np.reshape(np.argmax(y_test, axis=1), (1, -1))
    # print(norml_y_test)
    # accuracy = accuracy_score(np.reshape(np.argmax(y_test, axis=1), (1, -1)), final_predictions)
    accuracy = (np.sum(final_predictions == norml_y_test) / y_test.shape[0])
    print(f"Ensemble accuracy: {accuracy:.2%}")
    return accuracy

    if csv: 
        data = [["EEG", str(args.runs), str(args.epoch), str(args.learning_rate), str(results.history['val_accuracy'][-1])]]
        with open('testData.csv', "a") as file:
            writer = csv.writer(file)
            writer.writerows(data)
            print("write succeed")

    subject_score = subject_evaluate(model, x_test=x_test, y_test=y_test, person=person)

    print("==== subjects =====")
    print(subject_score)
    print("===================")

    return results.history['val_accuracy'][-1]


def subject_evaluate(model, x_test, y_test, person, verbose=False):
    '''
    Evaluate the model on each 9 subjects
    person contains subject label of each data in y_test
    '''
    subject_score = []
    for subject in np.unique(person).tolist():
        condition = person[:,0] == subject
        subject_y_test = y_test[condition]
        subject_x_test = x_test[condition]
        cnn_score = model.evaluate(subject_x_test, subject_y_test, verbose=verbose)  # TODO: delete this line before submitting
        subject_score.append(cnn_score[1])
        if verbose: print(f'Test accuracy of the subject {subject}: {cnn_score[1]}')

    return subject_score



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