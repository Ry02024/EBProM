import pandas as pd
import argparse
from Modules.machine_learning import *

def main(train_csv, validation_csv):
    """
    学習データと検証データのCSVファイルを読み込み、LightGBMでトレーニングを実施。
    
    Parameters:
    - train_csv: トレーニングデータのCSVファイルのパス
    - validation_csv: 検証データのCSVファイルのパス
    """
    # データ読み込み
    print("トレーニングデータを読み込んでいます...")
    train_df = pd.read_csv(train_csv)
    print("検証データを読み込んでいます...")
    validation_df = pd.read_csv(validation_csv)
    
    # データセットの作成
    print("データセットをセットアップしています...")
    lgb_train, lgb_eval = set_dataset(train_df, drop_num=0)  # validation_dfを直接渡します

    # モデルのトレーニング
    print("LightGBMでモデルをトレーニング中...")
    gbm = train_by_lightgbm_best_(lgb_train, lgb_eval)

    # モデルの保存
    gbm.save_model('lgbm_model.txt')
    print("モデルが 'lgbm_model.txt' として保存されました。")

if __name__ == "__main__":
    # コマンドライン引数からデータのパスを取得
    parser = argparse.ArgumentParser(description='LightGBM学習用スクリプト')
    parser.add_argument('--train_csv', type=str, required=True, help='トレーニングデータのCSVファイルのパス')
    parser.add_argument('--validation_csv', type=str, required=True, help='検証データのCSVファイルのパス')
    args = parser.parse_args()

    # main関数を実行
    main(args.train_csv, args.validation_csv)
