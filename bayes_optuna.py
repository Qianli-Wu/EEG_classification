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
    final_predictions = stats.mode(predictions, axis=0, keepdims=False)[0]

    return final_predictions

def keras_train(args, csv=False):
    '''
    Trains various deep learning models (CNN, Transformer, CNN+Transformer, and CNN+RNN) using Keras. 
    It supports ensemble training, early stopping, and model checkpointing. 
    The models are trained and validated using provided EEG data, split into training, validation, and testing sets.
    '''

    # Model parameters
    model_type = args.model
    learning_rate = args.learning_rate
    epochs = args.epoch
    time = args.time
    cnn_layers = args.cnn_layers
    transformer_layers = args.transformer_layers
    num_heads = args.num_heads         # Transformer Only
    patience = args.patience           # Early Stopping
    num_models = args.ensemble         # Model Ensembling
    optimizer = keras.optimizers.Adam(lr=learning_rate)

    x_train, x_valid, x_test, y_train, y_valid, y_test, person = load_data(time=time, onehot=True)


    models = []

    # Train multiple models
    for i in range(num_models):
        # Sadly, it's Python3.9, which does not support Structural Pattern Matching
        if model_type == 'cnn+transformer':
            model = cnn_transformer_model(input_shape=(int(time/2), 1, 22), num_cnn_layer=cnn_layers, filters=25, 
                                          num_heads=num_heads, num_transformer_layer=transformer_layers)
        elif model_type == 'cnn+rnn':
            model = cnn_rnn_model(input_shape=(int(time/2), 1, 22), num_cnn_layer=cnn_layers)
        elif model_type == 'transformer':
            model = transformer_model(input_shape=(int(time/2), 1, 22), num_transformer_layers=transformer_layers)
        elif model_type == 'cnn':
            model = cnn_model(input_shape=(int(time/2), 1, 22), num_cnn_layer=cnn_layers)
        else:
            model = cnn_transformer_model(input_shape=(int(time/2), 1, 22), num_cnn_layer=cnn_layers, filters=25, num_heads=num_heads)

        model.summary()

        # Compiling the model
        model.compile(loss='categorical_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
        
        # Add Callbacks: Checkpoint and EarlyStopping
        checkpoint = ModelCheckpoint(f'models/{model_type}_{cnn_layers}_{time}_{i}.h5', save_best_only=True)
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)


        # Training and validating the model
        results = model.fit(x_train,
                    y_train,
                    batch_size=64,
                    epochs=epochs,
                    callbacks=[callback, checkpoint],
                    validation_data=(x_valid, y_valid), verbose=True)
        models.append(model)


    
    # Model(s) Evaluation
    if num_models > 1:  # Ensemble Accuracy
        final_predictions = np.array(ensemble_predictions(models, x_test))  # 1772 x 1
        norml_y_test = np.reshape(np.argmax(y_test, axis=1), (1, -1))
        # accuracy = accuracy_score(np.reshape(np.argmax(y_test, axis=1), (1, -1)), final_predictions)
        accuracy = (np.sum(final_predictions == norml_y_test) / y_test.shape[0])
        subject_score = ensemble_subject_evaluate(models, x_test=x_test, y_test=y_test, person=person)
        print(f"Ensemble accuracy: {accuracy:.5%}")
    else:  # Single Model Accuracy
        accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
        subject_score = subject_evaluate(model, x_test=x_test, y_test=y_test, person=person)
        print(f"Model accuracy: {accuracy:.5%}")


    if csv: 
        data = [["EEG", str(args.runs), str(args.epoch), str(args.learning_rate), str(accuracy), *subject_score]]
        with open('testData.csv', "a") as file:
            writer = csv.writer(file)
            writer.writerows(data)
            print("write succeed")

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


def ensemble_subject_evaluate(models, x_test, y_test, person, verbose=False):
    '''
    Evaluate the ensembling models on each 9 subjects
    person contains subject label of each data in y_test
    '''
    subject_score = []
    for subject in np.unique(person).tolist():
        condition = person[:,0] == subject
        subject_y_test = y_test[condition]
        subject_x_test = x_test[condition]
        final_predictions = np.array(ensemble_predictions(models, subject_x_test))
        norml_y_test = np.reshape(np.argmax(subject_y_test, axis=1), (1, -1))
        accuracy = (np.sum(final_predictions == norml_y_test) / final_predictions.shape[0])
        subject_score.append(accuracy)
        if verbose: print(f'Test accuracy of the subject {subject}: {accuracy}')

    return subject_score


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
    # Define using TDP or GP
    if algo == "TPE":
        algo = optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24)
    # elif algo == "GP":
    #     from optuna.integration import SkoptSampler
    #     import skopt
    #     algo = SkoptSampler(skopt_kwargs={'base_estimator': 'GP',  # Gaussian Process
    #                                       'n_initial_points': 10,  # 10 initial points
    #                                       'acq_func': 'EI'})  # aqucition function

    # Create an instance for optimizer
    study = optuna.create_study(sampler=algo  , direction="minimize")
    # Start to optimize
    study.optimize(optuna_objective  # objective function
                   , n_trials=n_trials  # max number of trails
                   , show_progress_bar=True)


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