import pandas as pd
import argparse
from EBProM.Modules.utils import *

def main(data_dir):
    """
    データパスを引数として受け取り、前処理を実行する。
    
    Parameters:
    - data_dir: データが保存されているディレクトリのパス
    """
    # データ読み込み
    sales_history_df, item_categories_df, category_names_df, test_df = load_data(data_dir)
    # 関数load_dataを使って、過去の売上データ、商品カテゴリ、カテゴリ名、テストデータを読み込む

    # データの前処理
    join_data_df = preprocess_data(sales_history_df, item_categories_df, category_names_df)
    # 複数のデータセットを結合し、前処理を行う。欠損値処理や不要な列の削除などを実施。

    # 特徴量生成
    join_data_df_final = generate_features(join_data_df)
    # データから新しい特徴量（売上傾向や商品関連情報など）を生成する関数を実行

    # 商品カタログの生成（特徴量補完）
    catarog_df = complete_catalog(join_data_df_final)
    # カタログデータの作成。全ての商品情報を埋める処理を行う。

    # カタログのコピーを作成し、欠損値を補完する処理を実行
    catarog_df_copy = catarog_df.copy()  # オリジナルを保護するためのコピー
    catarog_fill = fill_missing_values(catarog_df_copy, test_df)  # 欠損値をテストデータを基に補完する

    # さらに特徴量の補完を行う
    catarog_copy = catarog_fill.copy()  # 欠損値補完後のデータをコピー
    catarog_copy_feats1 = fill_features(catarog_copy)  # 補完されたデータに追加の特徴量を埋める

    # スライディングウィンドウを使用してデータセットを生成
    catarog_feats1_copy = catarog_copy_feats1.copy()  # 特徴量補完後のコピーを作成
    train_df, test_df = generate_sliding_window_datasets(catarog_feats1_copy)
    # スライディングウィンドウを使用して、訓練データとテストデータを生成

    # 訓練データとテストデータに対して特徴量を生成
    train_df_gen, test_df_gen = generate_trend_features(train_df, test_df)
    # トレンドに関する特徴量を生成する関数を実行し、トレンドを捉えたデータに変換

    train_df_gen_copy = train_df_gen.copy()  # トレンド特徴量生成後の訓練データのコピー
    test_df_gen_copy = test_df_gen.copy()  # トレンド特徴量生成後のテストデータのコピー

    # カレンダー情報（曜日、祝日などの特徴量）を追加
    train_df_cal, test_df_cal = add_calendar_features(
        train_df=train_df_gen_copy,
        test_df=test_df_gen_copy,
        start_date='2018-01-01',
        end_date='2019-12-31',
        predict_year_month=(2019, 12)
    )
    # 関数add_calendar_featuresを使用して、データにカレンダーの特徴量（曜日や月の情報）を追加。期間は2018年1月1日から2019年12月31日まで。

    train_df_cal_copy = train_df_cal.copy()  # カレンダー情報追加後の訓練データをコピー
    test_df_cal_copy = test_df_cal.copy()  # カレンダー情報追加後のテストデータをコピー

    # 使用例として、グループ化するカラムや特徴量を設定
    group_cols = ['month_target', 'product_id']  # 月と商品IDでグループ化
    target_col = 'product_num'  # 目的変数（売上個数）
    feature_suffixes = [10, 9, 8]  # 使用する特徴量のサフィックス（後に続く数値）
    diff_features = {
        'ave_num_10_9': ('ave_num_10', 'ave_num_9'),
        'ave_num_10_8': ('ave_num_10', 'ave_num_8'),
    }
    # 過去の売上個数の差分などを特徴量として追加するための設定

    # 特徴量生成を関数で実行
    train_df_feats5, test_df_feats5 = feature_engineering(
        train_df_cal_copy,
        test_df_cal_copy,
        group_cols,
        target_col,
        feature_suffixes,
        diff_features=None  # 今回は差分特徴量は手動で追加するのでNone
    )

    train_df_feats5_copy = train_df_feats5.copy()  # 特徴量生成後の訓練データのコピー
    test_df_feats5_copy = test_df_feats5.copy()  # 特徴量生成後のテストデータのコピー

    # 売上が上昇傾向にあるかどうかのフラグを生成
    train_df_up, test_df_up = create_sales_uptrend_flag(train_df_feats5_copy, test_df_feats5_copy)
    # 上昇傾向のフラグ（売上が増加しているかどうか）を追加

    train_df_up_copy = train_df_up.copy()  # フラグ生成後の訓練データのコピー
    test_df_up_copy = test_df_up.copy()  # フラグ生成後のテストデータのコピー

    print('スライディングウィンドウの実行')
    # データ分割およびソート関数を使用して validation_df, train_df, sorted_test_df を作成
    validation_df, train_df, sorted_test_df = split_train_validation_and_sort_test(
        train_df_up=train_df_up_copy,
        test_df_feats5=test_df_up_copy,
        validation_main_flag=1,  # 検証用データにmain_flag=1を設定
        validation_month_target=12,  # 検証データは12ヶ月目のデータを使用
        sort_columns=['product_id', 'store_id']  # 商品IDと店舗IDでソート
    )

    # 処理後のデータを保存
    train_df.to_csv('train_df.csv', index=False)
    validation_df.to_csv('validation_df.csv', index=False)
    sorted_test_df.to_csv('sorted_test_df.csv', index=False)

    print("前処理が完了し、データが保存されました。")

if __name__ == "__main__":
    # コマンドライン引数からデータのディレクトリパスを取得
    parser = argparse.ArgumentParser(description='前処理のスクリプト')
    parser.add_argument('--data_dir', type=str, required=True, help='データが保存されているディレクトリのパス')
    args = parser.parse_args()

    # main関数を実行
    main(args.data_dir)
