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
    

# A, Bチームのランク・レベルを集約
def team_level_agg(input_data):
    
    A_rank_col = ["A1-rank", "A2-rank", "A3-rank", "A4-rank"]
    B_rank_col = ["B1-rank", "B2-rank", "B3-rank", "B4-rank"]
    rank_col = A_rank_col + B_rank_col
    
    A_level_col = ["A1-level", "A2-level", "A3-level", "A4-level"]
    B_level_col = ["B1-level", "B2-level", "B3-level", "B4-level"]
    sel_col = rank_col + A_level_col + B_level_col


    # 高ランク順に数字を当てはめ
    for col in rank_col:
        rk_order = {'c-' : 1, 'c' : 2, 'c+' : 3, 'b-' : 4,'b' : 5, 'b+' : 6, 'a-' : 7, 'a' : 8, 'a+' : 9, 's' : 10, 's+' : 11, 'x' : 12}
        input_data[col] = input_data[col].map(rk_order)
    

    output_data =pd.DataFrame()
    
    output_data["A_rank_mean"] = input_data[A_rank_col].mean(axis = 1)
    output_data["A_rank_std"]  = input_data[A_rank_col].std(axis = 1)
    output_data["A_rank_max"]  = input_data[A_rank_col].max(axis = 1)
    output_data["A_rank_min"]  = input_data[A_rank_col].min(axis = 1)

    output_data["B_rank_mean"] = input_data[B_rank_col].mean(axis = 1)
    output_data["B_rank_std"]  = input_data[B_rank_col].std(axis = 1)
    output_data["B_rank_max"]  = input_data[B_rank_col].max(axis = 1)
    output_data["B_rank_min"]  = input_data[B_rank_col].min(axis = 1)    
    
    
    output_data["A_level_mean"] = input_data[A_level_col].mean(axis = 1)
    output_data["A_level_std"]  = input_data[A_level_col].std(axis = 1)
    output_data["A_level_max"]  = input_data[A_level_col].max(axis = 1)
    output_data["A_level_min"]  = input_data[A_level_col].min(axis = 1)

    output_data["B_level_mean"] = input_data[B_level_col].mean(axis = 1)
    output_data["B_level_std"]  = input_data[B_level_col].std(axis = 1)
    output_data["B_level_max"]  = input_data[B_level_col].max(axis = 1)
    output_data["B_level_min"]  = input_data[B_level_col].min(axis = 1)
    
    output_data = output_data.fillna(0)
    
    return output_data


# A, Bチームのブキカテゴリーを指定してカウント
def buki_cate_count_agg(input_data, buki_cate):
    A_weapon_col = ["A1-weapon", "A2-weapon", "A3-weapon", "A4-weapon"]
    B_weapon_col = ["B1-weapon", "B2-weapon", "B3-weapon", "B4-weapon"]
    buki_col = A_weapon_col + B_weapon_col
     
    buki_data = pd.read_csv('./data/input/statink-weapon2.csv')
    target_buki_list    = list(buki_data[buki_data["category2"] == buki_cate]["key"].unique())
    no_target_buki_list = list(buki_data[buki_data["category2"] != buki_cate]["key"].unique())
    
    input_data = input_data[buki_col].replace(target_buki_list, 1).replace(no_target_buki_list, 0)
    
    output_data =pd.DataFrame()
    output_data[buki_cate + "_A_count"] =  input_data[A_weapon_col].sum(axis = 1)
    output_data[buki_cate + "_B_count"] =  input_data[B_weapon_col].sum(axis = 1)
    
    return output_data

# A, Bチームのスペシャルを指定してカウント
def special_cate_count_agg(input_data, buki_cate):
    A_weapon_col = ["A1-weapon", "A2-weapon", "A3-weapon", "A4-weapon"]
    B_weapon_col = ["B1-weapon", "B2-weapon", "B3-weapon", "B4-weapon"]
    buki_col = A_weapon_col + B_weapon_col
     
    buki_data = pd.read_csv('./data/input/statink-weapon2.csv')
    target_buki_list    = list(buki_data[buki_data["special"] == buki_cate]["key"].unique())
    no_target_buki_list = list(buki_data[buki_data["special"] != buki_cate]["key"].unique())
    
    
    output_data =pd.DataFrame()
    
    input_data = input_data[buki_col].replace(target_buki_list, 1).replace(no_target_buki_list, 0)
    
    output_data[buki_cate + "_A_count"] =  input_data[A_weapon_col].sum(axis = 1)
    output_data[buki_cate + "_B_count"] =  input_data[B_weapon_col].sum(axis = 1)
    
    return output_data


# ステージエリアを取得
def ft_stage_area(input_data):
    stage_data = pd.read_csv('./data/input/stagedata.csv')
    
    output_data = pd.DataFrame()
    output_data = pd.merge(input_data, stage_data, on = ["stage"], how = "left")["size"]
    output_data = pd.DataFrame(output_data)
    
    return output_data    

    
    
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


@cached_feature("ft_team_rank_train")
def create_train_team_level_agg(train, test):
    print("prepare ft_team_rank_train") 
    
    ft_team_rank_train = team_level_agg(train)
    ft_team_rank_test  = team_level_agg(test)

    return ft_team_rank_train

@cached_feature("ft_team_rank_test")
def create_test_team_level_agg(train, test):
    print("prepare ft_team_rank_train") 
    
    ft_team_rank_train = team_level_agg(train)
    ft_team_rank_test  = team_level_agg(test)

    return ft_team_rank_test


@cached_feature("ft_all_buki_cate_count_train")
def create_ft_all_buki_cate_count_train(train, test):
    print("prepare ft_all_buki_cate_count_train") 
    
    buki_data = pd.read_csv('./data/input/statink-weapon2.csv')
    buki_cate_list = list(buki_data["category2"].unique())

    train_base = pd.DataFrame()
    test_base  = pd.DataFrame()

    for buki_cate in buki_cate_list:
        train_tmp = pd.DataFrame()
        test_tmp  = pd.DataFrame()

        train_tmp = buki_cate_count_agg(train, buki_cate)
        test_tmp  = buki_cate_count_agg(test,  buki_cate)

        if train_base.empty:
            train_base = train_tmp
            test_base  = test_tmp

        else:
            train_base= pd.concat([train_base, train_tmp], axis=1)
            test_base = pd.concat([test_base,  test_tmp], axis=1)
    
    ft_all_buki_cate_count_train = train_base
    ft_all_buki_cate_count_test  = test_base
 
    return ft_all_buki_cate_count_train


@cached_feature("ft_all_buki_cate_count_test")
def create_ft_all_buki_cate_count_test(train, test):
    print("prepare ft_all_buki_cate_count_test") 
    
    buki_data = pd.read_csv('./data/input/statink-weapon2.csv')
    buki_cate_list = list(buki_data["category2"].unique())

    train_base = pd.DataFrame()
    test_base  = pd.DataFrame()

    for buki_cate in buki_cate_list:
        train_tmp = pd.DataFrame()
        test_tmp  = pd.DataFrame()

        train_tmp = buki_cate_count_agg(train, buki_cate)
        test_tmp  = buki_cate_count_agg(test,  buki_cate)

        if train_base.empty:
            train_base = train_tmp
            test_base  = test_tmp

        else:
            train_base= pd.concat([train_base, train_tmp], axis=1)
            test_base = pd.concat([test_base,  test_tmp], axis=1)
    
    ft_all_buki_cate_count_train = train_base
    ft_all_buki_cate_count_test  = test_base
 
    return ft_all_buki_cate_count_test


@cached_feature("ft_all_special_count_train")
def create_ft_all_special_count_train(train, test):
    print("prepare create_ft_all_special_count_train") 
    
    buki_data = pd.read_csv('./data/input/statink-weapon2.csv')
    special_cate_list = list(buki_data["special"].unique())

    train_base = pd.DataFrame()
    test_base  = pd.DataFrame()

    for special_cate in special_cate_list:
        train_tmp = pd.DataFrame()
        test_tmp  = pd.DataFrame()

        train_tmp = special_cate_count_agg(train, special_cate)
        test_tmp  = special_cate_count_agg(test,  special_cate)

        if train_base.empty:
            train_base = train_tmp
            test_base  = test_tmp

        else:
            train_base= pd.concat([train_base, train_tmp], axis=1)
            test_base = pd.concat([test_base,  test_tmp], axis=1)
 
    ft_all_special_count_train = train_base
    ft_all_special_count_test  = test_base    

    return ft_all_special_count_train


@cached_feature("ft_all_special_count_test")
def create_ft_all_special_count_test(train, test):
    print("prepare create_ft_all_special_count_test") 
    
    buki_data = pd.read_csv('./data/input/statink-weapon2.csv')
    special_cate_list = list(buki_data["special"].unique())

    train_base = pd.DataFrame()
    test_base  = pd.DataFrame()

    for special_cate in special_cate_list:
        train_tmp = pd.DataFrame()
        test_tmp  = pd.DataFrame()

        train_tmp = special_cate_count_agg(train, special_cate)
        test_tmp  = special_cate_count_agg(test,  special_cate)

        if train_base.empty:
            train_base = train_tmp
            test_base  = test_tmp

        else:
            train_base= pd.concat([train_base, train_tmp], axis=1)
            test_base = pd.concat([test_base,  test_tmp], axis=1)
 
    ft_all_special_count_train = train_base
    ft_all_special_count_test  = test_base    

    return ft_all_special_count_test


@cached_feature("ft_stagearea_train")
def create_train_stagearea(train, test):
    print("prepare ft_stagearea_train") 
    
    ft_stagearea_train = ft_stage_area(train)
    ft_stagearea_test  = ft_stage_area(test)

    return ft_stagearea_train

@cached_feature("ft_stagearea_test")
def create_test_stagearea(train, test):
    print("prepare ft_stagearea_test") 
    
    ft_stagearea_train = ft_stage_area(train)
    ft_stagearea_test  = ft_stage_area(test)

    return ft_stagearea_test


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
    
    create_train_team_level_agg(in_train, in_test)
    create_test_team_level_agg(in_train, in_test)
    
    create_ft_all_buki_cate_count_train(in_train, in_test)
    create_ft_all_buki_cate_count_test(in_train, in_test)
    
    create_ft_all_special_count_train(in_train, in_test)
    create_ft_all_special_count_test(in_train, in_test)
    
    create_train_stagearea(in_train, in_test)
    create_test_stagearea(in_train, in_test)
    
    