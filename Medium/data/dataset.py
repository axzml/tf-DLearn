#_*_ coding:utf-8 _*_
import numpy as np
import pandas as pd
import tensorflow as tf

from data.data_reader import *


def read_data(data_file, sep='\t'):
    df = pd.read_csv(data_file, sep=sep, header=None, encoding='utf-8')
    return df

def build_adult_dataset():
    train_file = '../Dataset/adult.data'
    test_file = '../Dataset/adult.test'

    CATEGORICAL_FEATURE_KEYS = [
        'workclass',
        'education',
        'education_num',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'gender',
        'native_country',
    ]
    NUMERIC_FEATURE_KEYS = [
        'age',
        'capital_gain',
        'capital_loss',
        'hours_per_week',
        'fnlwgt',
    ]

    LABEL_KEY = 'label'

    feature_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'label'
    ]

    data_frames = []
    label_map = {'<=50K': 0, '>50K': 1}
    for data_file in [train_file, test_file]:
        df = read_data(data_file, sep=',')
        df.columns = feature_names
        df['label'] = df['label'].map(label_map)
        data_frames.append(df)

    ignore_cols = [LABEL_KEY]
    ignore_cols += NUMERIC_FEATURE_KEYS
    fd = FeatureDictionary(dfTrain=data_frames[0], dfTest=data_frames[1],
                           numeric_cols=NUMERIC_FEATURE_KEYS, ignore_cols=ignore_cols)
    parser = DataParser(fd)

    datasets = []
    samples_num = []
    for frame in data_frames:
        Xi, Xv, y = parser.parse(df=frame, has_label=True)
        samples_num.append(len(Xi))
        dataset = tf.data.Dataset.from_tensor_slices((Xi, Xv, y))
        datasets.append(dataset)
    train_data, test_data = datasets
    params = {
        'feature_size': fd.feat_dim,
        'field_size': len(fd.feat_dict),
        'train_samples_num': samples_num[0],
        'test_samples_num': samples_num[1],
    }
    return train_data, test_data, params


if __name__ == '__main__':
    train_data, test_data, params = build_adult_dataset()
    print(params)
