---
layout: post
title: PLaMo translateでPDFのレイアウトを崩さずに翻訳 (mac OS)
date: 2025-06-08 10:33 +0900
description: ''
category: ''
tags: [PLaMo, LLM, PDFMathTranslate]
published: true
---


# PLaMo統合版PDFMathTranslate

サクッと読みたい論文は日本語が嬉しいのでClaude Codeに作ってもらいました。\
英語のPDFを入力すると、日本語のPDFと、日本語と英語の両方のPDFを出力します。\
Mlx-lmを使ってローカルでPLaMo translateを動かしています。\
Mac向けですが、mlx-lmで推論している部分を変えれば他のOSでも同じです。\
メモリは6~10 GBくらい使っているようです。\
macOSはSequoia15.4.1, M4のmacbookでメモリ16 GBで動作確認済みです。


![Example](/assets/img/2025_images/translate_example.png)


ソースコード:\
[https://github.com/shogo-d-nakamura/PDF-PLaMoTranslator](https://github.com/shogo-d-nakamura/PDF-PLaMoTranslator)


run.shにサンプルを示しています。\
`zsh install.sh` でvenv環境を作成したら、run.shのように -i で翻訳したいPDF or PDFが入ったディレクトリを指定します。\
PDFファイル(article.pdf)を指定した場合、article-out.pdf, article-dual.pdfが出力されます。\
ディレクトリ(articles)を指定した場合、articles-outを作成し、その中に{$FILENAME}-out.pdf, {$FILENAME}-dual.pdfを出力します。


```zsh
# for PDF
python translate.py -i ~/workspace/PLaMoTranslator/article.pdf

# for batch translation
python translate.py -i ~/workspace/PLaMoTranslator/articles
```


以下claudeが作成したREADMEの日本語訳です。




## クイックスタート

### 1. 依存関係のインストール

```zsh
# 仮想環境を作成し、PDFMathTranslateとMLX-LMをインストール
zsh install.sh
```

### 2. モデルのダウンロード
```zsh
python DL_plamo_translate.py
```

### 3. PDFの翻訳

```zsh
# テスト用サンプル
python translate_test_pdf.py
```

これにより`test-out.pdf`と`test-dual.pdf`が作成されます。

## 機能

### PLaMo統合機能
- **スマートトークン管理**: テキストを自動で約1024トークンにチャンク分割
- **文境界対応分割**: チャンク分割時に文の境界を保持
- **高品質翻訳**: mlx-community/plamo-2-translateモデルを使用
- **レイアウト保持**: PDFMathTranslateを使用してオリジナルのPDFフォーマットを維持

### 自動機能
- **トークン推定**: 約4文字で1トークンとして推定
- **エラーハンドリング**: 翻訳失敗時の適切なフォールバック
- **進捗追跡**: 大きなドキュメントの翻訳進捗を表示
- **フォーマット対応**: 数式や複雑なレイアウトに対応

## 使用例

### 基本的な翻訳スクリプト

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, 'PDFMathTranslate')
from pdf2zh import translate

# PLaMoを使用してPDFを翻訳
translate(
    files=["input.pdf"],
    output="./output",
    lang_in="en",        # 英語入力
    lang_out="ja",       # 日本語出力
    service="plamo"      # PLaMo翻訳を使用
)
```

### コマンドライン使用

```zsh
# PDFMathTranslateディレクトリに移動
cd PDFMathTranslate

# PLaMoサービスを使用して翻訳
python -m pdf2zh.cli \
    --service plamo \
    --lang-in en \
    --lang-out ja \
    --output ../output \
    ../test.pdf
```

### 高度な使用例

```python
from pdf2zh import translate

# カスタム設定
translate(
    files=["document.pdf"],
    output="./translations",
    lang_in="en",
    lang_out="ja", 
    service="plamo:custom-model-name",  # カスタムモデルを使用
    thread=1,                           # PLaMo用シングルスレッド
    ignore_cache=True,                  # 新規翻訳
    envs={
        "PLAMO_MODEL": "mlx-community/plamo-2-translate",
        "PLAMO_MAX_TOKENS": "1024"
    }
)
```

## 設定

### 環境変数

PLaMoの動作をカスタマイズするために以下の環境変数を設定：

```zsh
export PLAMO_MODEL="mlx-community/plamo-2-translate"
export PLAMO_MAX_TOKENS="1024"
```

### モデル設定

```python
# 異なるPLaMoモデルを使用
service_options = [
    "plamo",                                    # デフォルトモデル
    "plamo:mlx-community/plamo-2-translate",   # 明示的なモデル指定
    "plamo:custom-plamo-model"                 # カスタムモデル
]
```

## 動作原理

### トークン管理プロセス

1. **テキスト抽出**: PDFページからテキストを抽出
2. **トークン推定**: トークン数を推定（約4文字 = 1トークン）
3. **スマートチャンク分割**: 文の境界でテキストを分割
4. **翻訳**: MLX-LM経由でPLaMoを使用して各チャンクを翻訳
5. **再構築**: レイアウトを保持して翻訳テキストを再結合