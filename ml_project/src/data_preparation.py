# module data_preparation

import pandas as pd
import numpy as np

PATH = 'data/'


def train_test_split(X, test_size=0.3, random_state=42):
    np.random.seed(seed=random_state)
    indx = np.random.permutation(len(X))
    threshold = int(test_size * len(X))
    indx_train, indx_test = indx[threshold:], indx[:threshold]
    
    return X.iloc[indx_train], X.iloc[indx_test]


def split_data_on_x_y(frame, name):
    target = np.where(frame[1].values == 'M', 1, 0).reshape(-1, 1)
    Y = np.hstack([target, 1 - target])
    feature = frame.columns[2:]

    frame[feature].to_csv(PATH + f'{name}.csv', index=False)
    pd.DataFrame(Y).to_csv(PATH + f'{name}_target.csv', index=False)
    
    
def main():
    data = pd.read_csv(PATH + 'data.csv', names=range(32))

    train, test = train_test_split(data)
    split_data_on_x_y(train, 'train')     
    split_data_on_x_y(test, 'test')
    
    
if __name__ == '__main__':
    main()
