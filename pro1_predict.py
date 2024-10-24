import pandas as pd
import argparse
from lightgbm import Booster

def main(model_path, test_csv):
    """
    予測モデルとテストデータを読み込み、予測を実施。

    Parameters:
    - model_path: 予測モデルのパス
    - test_csv: テストデータのCSVファイルのパス
    """
    # モデルの読み込み
    gbm = Booster(model_file=model_path)
    # テストデータの読み込み
    test_df = pd.read_csv(test_csv)
    # 予測
    final_predicted = gbm.predict(test_df)
    # numpy.ndarray を DataFrame に変換
    final_predicted_df = pd.DataFrame(final_predicted, columns=['Predicted'])

    # CSV に保存
    final_predicted_df.to_csv("EBProM/Data/Preprocess/final_predicted.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='予測用スクリプト')
    parser.add_argument('--model_path', type=str, required=True, help='予測モデルのパス')
    parser.add_argument('--test_csv', type=str, required=True, help='テストデータのCSVファイルのパス')
    args = parser.parse_args()
    main(args.model_path, args.test_csv)
    print("予測が完了しました。")
