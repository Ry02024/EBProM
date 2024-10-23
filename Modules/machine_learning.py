import pandas as pd
import random
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping

# Xとyを分割し、学習データと検証データに分割してランダムに特徴量を削除する関数
def learn_valid_split(df, test_size=0.2, random_state=42, drop_num=0):
    """
    Xとyに分割し、学習データと検証データに分割した上で、指定された数の特徴量をランダムに削除。

    Parameters:
    - df: データフレーム (全体のデータ)
    - test_size: 検証データの割合 (default: 0.2)
    - random_state: 乱数シード (default: 42)
    - drop_num: 削除する特徴量の数 (default: 0)

    Returns:
    - train_x_df: 学習用の特徴量データフレーム
    - train_y_df: 学習用のターゲットデータフレーム
    - validation_x_df: 検証用の特徴量データフレーム
    - validation_y_df: 検証用のターゲットデータフレーム
    """
    # 特徴量とターゲットに分割
    x_df, y_df = split_x_y(df)

    # 学習データと検証データに分割
    train_x_df, validation_x_df, train_y_df, validation_y_df = manual_train_test_split(
        x_df, y_df, test_size=test_size, random_state=random_state)

    # ランダムに指定された数だけ特徴量を削除
    if drop_num > 0:
        random.seed(random_state)
        drop_idx = [random.randint(0, len(train_x_df.columns) - 1) for _ in range(drop_num)]
        drop_columns = train_x_df.columns[drop_idx]
        train_x_df = train_x_df.drop(drop_columns, axis=1)
        validation_x_df = validation_x_df.drop(drop_columns, axis=1)

    return train_x_df, train_y_df, validation_x_df, validation_y_df

# データセットをセットアップするシンプルな関数
def set_dataset(df, test_size=0.2, random_state=42, drop_num=0):
    """
    データをセットアップし、学習データと検証データを生成して、LightGBM用に変換。

    Parameters:
    - df: データフレーム (全体のデータ)
    - test_size: 検証データの割合 (default: 0.2)
    - random_state: 乱数シード (default: 42)
    - drop_num: 削除する特徴量の数 (default: 0)

    Returns:
    - lgb_train: LightGBMのトレーニングデータ
    - lgb_eval: LightGBMの評価データ
    """
    # 学習データと検証データを分割し、ランダム特徴量削除も同時に実行
    train_x_df, train_y_df, validation_x_df, validation_y_df = learn_valid_split(
        df, test_size=test_size, random_state=random_state, drop_num=drop_num)

    # LightGBM用のデータセットに変換
    lgb_train = lgb.Dataset(train_x_df, train_y_df)
    lgb_eval = lgb.Dataset(validation_x_df, validation_y_df)

    return lgb_train, lgb_eval

# 特徴量とターゲットに分割する関数
def split_x_y(df):
    """
    特徴量とターゲットを分割する。

    Parameters:
    - df: データフレーム

    Returns:
    - x_df: 特徴量データフレーム
    - y_df: ターゲットデータフレーム
    """
    # 特徴量とターゲットに分割
    x_df = df.drop(['product_price_11', 'product_price_12', 'product_num_11', 'product_num_12'], axis=1)
    y_df = df['product_num_12']

    return x_df, y_df

# 学習データと検証データを分割するための関数
def manual_train_test_split(x_df, y_df, test_size=0.2, random_state=42):
    """
    特徴量データとターゲットデータを学習データと検証データに分割する。

    Parameters:
    - x_df: 特徴量データフレーム
    - y_df: ターゲットデータフレーム
    - test_size: 検証データの割合
    - random_state: 乱数シード

    Returns:
    - train_x_df: 学習用の特徴量データ
    - validation_x_df: 検証用の特徴量データ
    - train_y_df: 学習用のターゲットデータ
    - validation_y_df: 検証用のターゲットデータ
    """
    # データ分割
    split_index = int((1 - test_size) * len(x_df))
    train_x_df = x_df.iloc[:split_index].reset_index(drop=True)
    validation_x_df = x_df.iloc[split_index:].reset_index(drop=True)
    train_y_df = y_df.iloc[:split_index].reset_index(drop=True)
    validation_y_df = y_df.iloc[split_index:].reset_index(drop=True)

    return train_x_df, validation_x_df, train_y_df, validation_y_df

# LightGBMでのモデルトレーニング関数
def train_by_lightgbm_best_(lgb_train, lgb_eval):
    """
    LightGBMを使ってモデルをトレーニングする関数。

    Parameters:
    - lgb_train: トレーニングデータセット
    - lgb_eval: 検証データセット

    Returns:
    - gbm: トレーニングされたLightGBMモデル
    """
    # ハイパーパラメータを設定
    params = {'metric': 'rmse'}

    # コールバック関数の設定
    callbacks = [
        log_evaluation(period=500),  # 500イテレーションごとに結果を表示
        early_stopping(stopping_rounds=500, verbose=True)
    ]

    # モデルのトレーニング
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=[lgb_train, lgb_eval],
                    num_boost_round=10000,
                    callbacks=callbacks)
    return gbm
