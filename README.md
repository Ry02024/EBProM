# EBProM

このプロジェクトは、**商品売上の予測**を行うためのデータセットとモジュールを使用して、機械学習モデルを構築し、シミュレーションを行うPythonベースのリポジトリです。

### ディレクトリ構成

```
Data/                   # データセットを含むフォルダ
  ├── category_names.csv         # 商品カテゴリ名のデータ
  ├── item_categories.csv        # 商品カテゴリに関するデータ
  ├── sales_history.csv          # 売上履歴データ
  ├── sample_submission.csv      # サンプル提出用フォーマット
  ├── test.csv                   # テスト用データ
Modules/               # 機械学習とシミュレーションのスクリプト
  ├── machine_learning.py        # 機械学習モデルのトレーニングと評価に関するスクリプト
  ├── simulation.py              # シミュレーションを行うためのスクリプト
  ├── utils.py                   # ユーティリティ関数（データ処理など）
.github/                # GitHub Actionsワークフロー設定ファイル
  └── workflows/        
      └── main.yml               # 自動化のフローレシピ（GitHub Actions用のYAMLファイル）
EBProM.ipynb           # プロジェクトの実行に使用されるJupyterノートブック
README.md              # リポジトリの説明と使用方法
requirements.txt
```

### 説明:
- **`.github/workflows/main.yml`**: GitHub Actionsでプロジェクトの自動化フローを実行するための設定ファイルです。このファイルで、データの前処理、モデルのトレーニング、シミュレーション、結果のコミット・プッシュを自動化します。

これにより、`README.md`内でプロジェクトのディレクトリ構造に`workflow`が含まれ、GitHub Actionsの自動化フローが明確に説明されます。

## ファイル詳細

### データセット (`Data/`)

- **category_names.csv**  
  商品カテゴリの名前が含まれています。

- **item_categories.csv**  
  商品カテゴリごとのデータです。

- **sales_history.csv**  
  売上履歴データが含まれており、機械学習モデルの訓練に使用されます。

- **sample_submission.csv**  
  モデルの結果を提出する際のサンプルフォーマットです。

- **test.csv**  
  テストデータとして使用されるデータセットです。

### モジュール (`Modules/`)

- **machine_learning.py**  
  機械学習モデルの訓練と評価に使用されるスクリプトです。このスクリプトではLightGBMなどのモデルが使われ、売上予測を行います。

- **simulation.py**  
  商品や売上に対するシミュレーションを実行するためのスクリプトです。

- **utils.py**  
  データの前処理や補助的なタスクに使用されるユーティリティ関数が含まれています。

自動化のフローをREADMEの1項目として追加する場合、次のように簡潔に記述できます。

### 自動化のフロー（`.github/workflows/`）

このプロジェクトでは、GitHub Actionsを使用して自動的にデータの前処理、機械学習モデルのトレーニング、シミュレーションの実行、結果のコミット・プッシュが行われます。

1. **リポジトリのクローンとセットアップ**  
   リポジトリをクローンし、必要なPythonライブラリが自動でインストールされます。

2. **データの前処理**  
   `Data/`フォルダ内のデータを前処理し、指定したフォルダに保存されます。

3. **機械学習モデルのトレーニング**  
   前処理済みのデータを使用して、LightGBMを使った機械学習モデルが自動でトレーニングされます。

4. **シミュレーション**  
   モデルを使用して売上予測シミュレーションを実行します。

5. **結果のコミットとプッシュ**  
   前処理済みデータとシミュレーション結果が自動的にリポジトリにコミット・プッシュされます。

自動化は、データがプッシュされた際やスケジュールに基づいて実行されます。

### その他

- **EBProM.ipynb**  
  プロジェクトの主要な実行を行うJupyterノートブック。機械学習モデルの訓練、予測、結果の可視化を行うコードが含まれています。

- **README.md**  
  現在ご覧になっているファイルです。このリポジトリの全体的な説明や使用方法を記述しています。

## 使用方法

1. **データの準備**  
   `Data/`フォルダにデータセットを配置してください。

2. **ノートブックの実行**  
   Jupyterノートブック (`EBProM.ipynb`) を実行することで、機械学習モデルのトレーニング、評価、予測が行えます。

3. **モジュールのカスタマイズ**  
   必要に応じて、`Modules/`フォルダ内のスクリプトを編集し、カスタム処理を追加してください。
   
## 参考
AI Quest 2020 (経産省、SIGNATE)：KUBOKOさんのコード

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Copyright
© 2024 [Ry02024]. All Rights Reserved.

## 貢献

もしこのリポジトリに貢献したい場合は、プルリクエストを歓迎します。また、バグ報告やフィードバックがあれば [Issues](https://github.com/ユーザー名/リポジトリ名/issues) ページに投稿してください。

---

このREADMEはプロジェクトの基本的な構造と使用方法を簡潔に説明するものです。
