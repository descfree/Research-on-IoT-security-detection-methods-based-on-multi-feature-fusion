import tensorflow
import pandas as pd
import numpy as np
import os
from time import time

from keras.layers import Dense, Dropout, RNN, LSTM, GRU
from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint

from kdd_processing import kdd_encoding

from results_visualisation import print_results
from keras.layers import SimpleRNN, LSTM, GRU

config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.compat.v1.Session(config=config)

csv_values = ['epochs', 'acc', 'loss', 'val_acc', 'val_loss', "train_data",
              "features_nb", 'loss_fct', 'optimizer', 'activation_fct',
              'layer_nb', 'unit_nb', 'batch_size', 'dropout', 'cell_type',
              'encoder']

csv_best_res = ['param', 'value', 'min_mean_val_loss']



params = {'epochs': 3, 'train_data': 484021, 'features_nb': 7,
          'loss_fct': 'mse', 'optimizer': 'rmsprop',
          'activation_fct': 'sigmoid', 'layer_nb': 2, 'unit_nb': 64,
          'batch_size': 1024, 'dropout': 0.5, 'cell_type': 'LSTM',
          'encoder': 'labelencoder', 'training_nb': 1,
          'resultstocsv': True, 'resultstologs': True, 'showresults': True,
          'shuffle': True}

params_var = {'encoder': [
                          'labelencoder'],
              }

model_path = 'D:/Research on IoT security detection methods based on multi-feature fusion/Research on IoT security detection methods based on multi-feature fusion/models/'
logs_path = 'D:/Research on IoT security detection methods based on multi-feature fusion/Research on IoT security detection methods based on multi-feature fusion/logs/'
res_path = 'D:/Research on IoT security detection methods based on multi-feature fusion/Research on IoT security detection methods based on multi-feature fusion/results/' + 'testcsv/'

if params['resultstologs'] is True:
    res_name = params['cell_type'] + '_' + str(params['features_nb']) +\
        '_' + str(params['train_data']) + '_' + params['optimizer'] + '_' +\
        params['activation_fct'] + '_' + str(params['layer_nb']) + '_' +\
        str(params['unit_nb']) + '_' + str(params['batch_size']) + '_' +\
        str(params['dropout']) + '_' + params['cell_type'] + '_' +\
        params['encoder'] + '_' + str(time())



def load_data():

    x_train, x_test, y_train, y_test = kdd_encoding(params)



    x_train = np.array(x_train).reshape([-1, x_train.shape[1], 1])
    x_test = np.array(x_test).reshape([-1, x_test.shape[1], 1])
    return x_train, x_test, y_train, y_test



def train_model(x_train, x_test, y_train, y_test):
    if params['cell_type'] == 'RNN':
        cell = SimpleRNN
    elif params['cell_type'] == 'LSTM':
        cell = LSTM
    elif params['cell_type'] == 'GRU':
        cell = GRU

    model = Sequential()

    for _ in range(params['layer_nb']-1):
        model.add(cell(units=params['unit_nb'],
                       input_shape=(x_train.shape[1:]), return_sequences=True))
        model.add(Dropout(rate=params['dropout']))


    if params['layer_nb'] == 1:
        model.add(cell(units=params['unit_nb'], input_shape=x_train.shape[1:]))
        model.add(Dropout(rate=params['dropout']))
    else:
        model.add(cell(units=params['unit_nb']))
        model.add(Dropout(rate=params['dropout']))

    model.add(Dense(units=y_train.shape[1],
                    activation=params['activation_fct']))

    model.compile(loss=params['loss_fct'], optimizer=params['optimizer'],
                  metrics=['accuracy'])

    if params['resultstologs'] is True:
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        save_model = ModelCheckpoint(filepath=model_path + res_name,
                                     monitor='val_acc', save_best_only=True)
        tensorboard = TensorBoard(logs_path+res_name)
        callbacks = [save_model, tensorboard]
    else:
        callbacks = None

    model.summary()

    hist = model.fit(x_train, y_train, params['batch_size'], params['epochs'],
                     verbose=1, shuffle=params['shuffle'],
                     validation_data=(x_test, y_test), callbacks=callbacks)

    if params['showresults'] is True:
        print_results(params, model, x_train, x_test, y_train, y_test)

    return hist


def res_to_csv(x_train, x_test, y_train, y_test):
    ref_min_val_loss = 10
    nsmall = 5  #

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    full_res_path = res_path + 'full_results.csv'
    best_res_path = res_path + 'best_result.csv'


    results_df = pd.DataFrame(columns=csv_values)
    results_df.to_csv(full_res_path, index=False)

    best_res_df = pd.DataFrame(columns=csv_best_res)

    def fill_dataframe(df, history, epoch):
        df = df.append({'epochs': epoch,
                        'acc':  history.history['acc'][epoch],
                        'loss': history.history['loss'][epoch],
                        'val_acc': history.history['val_acc'][epoch],
                        'val_loss': history.history['val_loss'][epoch],
                        'train_data': params['train_data'],
                        'features_nb': params['features_nb'],
                        'loss_fct': params['loss_fct'],
                        'optimizer': params['optimizer'],
                        'activation_fct': params['activation_fct'],
                        'layer_nb': params['layer_nb'],
                        'unit_nb': params['unit_nb'],
                        'batch_size': params['batch_size'],
                        'dropout': params['dropout'],
                        'cell_type': params['cell_type'],
                        'encoder': params['encoder']},
                       ignore_index=True)
        return df


    def min_mean_val_loss(feature):

        df = pd.read_csv(res_path+feature+".csv", index_col=False)
        names = df[feature].unique().tolist()
        df_loss = pd.DataFrame(columns=names)


        for i in range(len(names)):
            df_value_loss = df.loc[df[feature] == names[i]]
            df_value_loss = df_value_loss.nsmallest(nsmall, 'val_loss')
            df_loss[names[i]] = np.array(df_value_loss['val_loss'])


        return df_loss.mean().idxmin(), df_loss.mean().min()

    for feature in params_var.keys():
        results_df.to_csv(res_path + feature + ".csv", index=False)
        save_feature_value = params[feature]

        print(params_var[feature])
        for feature_value in params_var[feature]:
            df_value = pd.DataFrame(columns=csv_values)
            params[feature] = feature_value

            print(feature_value)



            for _ in range(params['training_nb']):
                history = train_model(x_train, x_test, y_train, y_test)


                for epoch in range(params['epochs']):
                    df_value = fill_dataframe(df_value, history, epoch)

            df_value.to_csv(full_res_path, header=False, index=False, mode='a')
            df_value.to_csv(res_path + feature + ".csv", header=False,
                            index=False, mode='a')

        feature_value_min_loss, min_mean_loss = min_mean_val_loss(feature)


        if min_mean_loss < ref_min_val_loss:
            params[feature] = feature_value_min_loss
            ref_min_val_loss = min_mean_loss
        else:
            params[feature] = save_feature_value


        best_res_df = best_res_df.append({'param': feature,
                                          'value': params[feature],
                                          'min_mean_val_loss': min_mean_loss},
                                         ignore_index=True)
        best_res_df.to_csv(best_res_path, index=False)
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()

    for i in range(params['training_nb']):
        if params['resultstocsv'] is False:
            train_model(x_train, x_test, y_train, y_test)
        else:
            res_to_csv(x_train, x_test, y_train, y_test)
