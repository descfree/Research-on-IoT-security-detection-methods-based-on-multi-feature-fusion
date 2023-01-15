from sklearn.preprocessing import (StandardScaler, OrdinalEncoder,
                                   LabelEncoder, MinMaxScaler, OneHotEncoder)
from keras.utils import to_categorical
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


full_features = ["network_security", "authentication", " access_control", "cipher", "node", "non_reputation", "router", "label"]



entry_type ={'normal': 'normal',
              'key': ['keysecurity.', 'kf.', 'kau.',
                        'kne.','kac.','kno.','knr.'],
              'f': ['fr.', 'fac.', 'fno.', 'fnr.','fk.','fau.','fne.'],
              'au': ['authentication.', 'auf.', 'aune.', 'auac.','auk.','auno.','aunr.'],
              'nec': ['nes.', 'nesf.', 'nesac.', 'nesnr.','nesk.','nesau.','nesno.'],
              'ac': ['acl.', 'aclk.', 'aclno.', 'aclnr.','aclf.','aclau.','aclne.'],
             'no': ['node.', 'nodek.', 'nodeau.', 'nodene.','nodef.','nodenr.','nodeac.'],
             'nor': ['nr.', 'nrk.', 'nrau.', 'nrno.','nrf.','nrne.','nrac.']}







def kdd_encoding(params):

    data_path = "D:/Research on IoT security detection methods based on multi-feature fusion/Research on IoT security detection methods based on multi-feature fusion/data/"

    train_data_path = data_path+"train.csv"
    test_data_path = data_path + "test.csv"

    full_features.append("difficulty")


    train_df = pd.read_csv(train_data_path, names=full_features)
    test_df = pd.read_csv(test_data_path, names=full_features)

    def process_dataframe(df):


        features = full_features

        df = df[features]


        df['label'] = df['label'].replace(['normal.', 'normal'], 0)
        for i in range(len(entry_type['key'])):
            df['label'] = df['label'].replace(
                [entry_type['key'][i], entry_type['key'][i][:-1]], 1)
        for i in range(len(entry_type['f'])):
            df['label'] = df['label'].replace(
                [entry_type['f'][i], entry_type['f'][i][:-1]], 2)
        for i in range(len(entry_type['au'])):
            df['label'] = df['label'].replace(
                [entry_type['au'][i], entry_type['au'][i][:-1]], 3)
        for i in range(len(entry_type['nec'])):
            df['label'] = df['label'].replace(
                [entry_type['nec'][i], entry_type['nec'][i][:-1]], 4)
        for i in range(len(entry_type['ac'])):
            df['label'] = df['label'].replace(
                [entry_type['ac'][i], entry_type['ac'][i][:-1]], 5)
        for i in range(len(entry_type['no'])):
            df['label'] = df['label'].replace(
                [entry_type['no'][i], entry_type['no'][i][:-1]], 6)
        for i in range(len(entry_type['nor'])):
            df['label'] = df['label'].replace(
                [entry_type['nor'][i], entry_type['nor'][i][:-1]], 7)




        if "difficulty" in df.columns:
            df = df.drop(columns='difficulty')


        y = df['label']
        x = df.drop(columns='label')


        if params['encoder'] == 'ordinalencoder':
            x = OrdinalEncoder().fit_transform(x)

        elif params['encoder'] == 'labelencoder':
            x = x.apply(LabelEncoder().fit_transform)



            if params['encoder'] == "standardscaler":
                x = StandardScaler().fit_transform(x)

            elif params['encoder'] == "minmaxscaler01":
                x = MinMaxScaler(feature_range=(0, 1)).fit_transform(x)

            elif params['encoder'] == "minmaxscaler11":
                x = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x)
        return x, y

    x_train, Y_train = process_dataframe(train_df)
    x_test, Y_test = process_dataframe(test_df)

    y_train = to_categorical(Y_train)
    y_test = to_categorical(Y_test)
    print(x_train, x_test, y_train, y_test)
    return x_train, x_test, y_train, y_test
