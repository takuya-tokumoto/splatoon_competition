import pandas as pd
import feather

def load_datasets(feats, target_name):
    dfs = [feather.read_dataframe(f'features/{f}_train.f') for f in feats]
    train = pd.concat(dfs, axis=1, sort=False)
    
#     今回は目的変数も含んでいるので削除
    X_train = train.drop([target_name], axis = 1)
    
    dfs = [feather.read_dataframe(f'features/{f}_test.f') for f in feats]
    X_test = pd.concat(dfs, axis=1, sort=False)
    
    return X_train, X_test


def load_target(target_name):
    train = pd.read_csv('./data/input/train_data.csv')
    y_train = train[target_name]
    return y_train