# CNN-Baseline

Pytorch Lightningを使用した画像分類モデルベースライン

```
pytorch-lightning==0.9.0
torch==1.6.0
```

## データ

- input/フォルダを作成し、下記のようなフォルダ構成を想定

サンプルデータは[こちら](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)を使用

```bash
input
├── test           # テストデータ
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   └── 5.jpg
├── test.csv       # テストデータのアノテーション（image_id）
├── train          # 学習&検証データ
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   ├── cat.2.jpg
│   ├── cat.3.jpg
└── train.csv      # 学習&検証データのアノテーション（image_id, target）
```

## ロギング

[Comet_ml](https://www.comet.ml/site/)を使用

- config.ymlの該当箇所にComet_mlのAPI_KEYとPROJECT_NAMEを記入する
  
  
## 環境構築

- poetryのインストール

- poetry.lockからライブラリをインストール

以下コマンドを実行

```bash
poetry install
```

## 学習方法

以下コマンドを実行

```bash
poetry run python train.py
```

バッチサイズを32、学習率を0.001で学習する場合

```bash
poetry run python train.py train.batch_size=32 train.lr=0.001
```
