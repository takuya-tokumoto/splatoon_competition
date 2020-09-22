from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from nyaggle.feature_store import cached_feature
from tqdm import tqdm

INPUT_DIR = "data/input/"


# 欠損値は、全て「－１」とする。
def fill_all_null(df):
    for col_name in df.columns[df.isnull().sum()!=0]:
        df[col_name] = df[col_name].fillna(-1)
    
        
# ターゲットエンコーディングの関数定義
def change_to_target2(train_df,test_df,input_column_name,output_column_name):
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=True, random_state=71)
    #=========================================================#
    c=input_column_name
    # 学習データ全体で各カテゴリにおけるyの平均を計算
    data_tmp = pd.DataFrame({c: train_df[c],'target':train_df['y']})
    target_mean = data_tmp.groupby(c)['target'].mean()
    #テストデータのカテゴリを置換
    test_df[output_column_name] = test_df[c].map(target_mean)
    
    # 変換後の値を格納する配列を準備
    tmp = np.repeat(np.nan, train_df.shape[0])

    for i, (train_index, test_index) in enumerate(kf.split(train_df)): # NFOLDS回まわる
        #学習データについて、各カテゴリにおける目的変数の平均を計算
        target_mean = data_tmp.iloc[train_index].groupby(c)['target'].mean()
        #バリデーションデータについて、変換後の値を一時配列に格納
        tmp[test_index] = train_df[c].iloc[test_index].map(target_mean) 
        

    #変換後のデータで元の変数を置換
    train_df[output_column_name] = tmp
    #========================================================#   

        
@cached_feature("ft_engineered_train")
def create_train_ft_engineered_mart(train, test):
    print("prepare ft_engineered_train_mart")

    # オブジェクトの列のリストを作成
    object_col_list = train.select_dtypes(include=object).columns
    
    # オブジェクトの列は全てターゲットエンコーディング実施
    for col in object_col_list:
        change_to_target2(train,test,col,"enc_"+col)
    
    #　変換前の列を削除
    _train = train.drop(object_col_list,axis=1)
    _test = test.drop(object_col_list,axis=1)
    
    #　'id'の列を削除
    ft_engineered_train = _train.drop('id',axis=1)
    ft_engineered_test = _test.drop('id',axis=1)
    
    #  target encordingで紐づけれないデータは欠損するので-1で埋める
    ft_engineered_train = ft_engineered_train.fillna(-1)
    ft_engineered_test  = ft_engineered_test.fillna(-1)
    
    return ft_engineered_train


@cached_feature("ft_engineered_test")
def create_test_ft_engineered_mart(train, test):
    print("prepare ft_engineered_train_mart") 
    
    # オブジェクトの列のリストを作成
    object_col_list = train.select_dtypes(include=object).columns
    
    # オブジェクトの列は全てターゲットエンコーディング実施
    for col in object_col_list:
        change_to_target2(train,test,col,"enc_"+col)
    
    #　変換前の列を削除
    _train = train.drop(object_col_list,axis=1)
    _test = test.drop(object_col_list,axis=1)
    
    #　'id'の列を削除
    ft_engineered_train = _train.drop('id',axis=1)
    ft_engineered_test = _test.drop('id',axis=1)
    
    #  target encordingで紐づけれないデータは欠損するので-1で埋める
    ft_engineered_train = ft_engineered_train.fillna(-1)
    ft_engineered_test  = ft_engineered_test.fillna(-1)
    
    
    return ft_engineered_test


if __name__ == "__main__":
    ROOT_DIR = ""
    INPUT_DIR = ROOT_DIR + "data/input/"
    in_train = pd.read_feather(INPUT_DIR + "train_data.feather")
    in_test = pd.read_feather(INPUT_DIR + "test_data.feather")
    
    # 訓練データ、テストデータの欠損値を「－１」で補完
    fill_all_null(in_train)
    fill_all_null(in_test) 

    create_train_ft_engineered_mart(in_train, in_test)
    create_test_ft_engineered_mart(in_train, in_test)