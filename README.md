# Ingredient Normalization Project

材料名の表記ゆれ・類義語を統一するための研究用プロジェクトです。  
Sentence-BERT / CrossEncoder / 文字類似度 / 読み仮名特徴を組み合わせて、
自動クラスタリングと代表語選択を行います。

---

## 📌 概要

本プロジェクトでは、レシピデータに含まれる材料名の揺れ（例：  
「しょうゆ / 醤油 / 正油」など）を自動的に統一することを目的としています。

主に以下の手法を組み合わせています。

- Sentence-BERT による意味類似度
- Cross-Encoder による再ランキング
- Jaro-Winkler による文字列類似度
- 読み仮名・語尾特徴による補助スコア
- グラフクラスタリングによる統合

## アルゴリズム概要

1. 材料名の正規化  
   全角・半角や記号の違いをそろえ、比較しやすい形に変換する。

2. 読み仮名の取得  
   「醤油」「しょうゆ」のように表記は違っても読みが近い材料を比較できるように、SudachiPyで読みを取得する。

3. 候補ペアの絞り込み  
   全ての材料ペアを比較すると計算量が大きくなるため、Sentence-BERTの近傍探索や読みの先頭文字、語尾特徴を使って、統一候補となるペアを先に絞り込む。

4. 類似度スコアの計算  
   Cross-Encoder、Sentence-BERTのコサイン類似度、Jaro-Winkler、読み仮名・語尾特徴を組み合わせて、同一材料かどうかを判定する。

5. グラフクラスタリング  
   しきい値以上のペアを辺としてグラフを作成し、連結成分ごとに同一材料の候補としてまとめる。

6. 代表語の決定  
   クラスタ内で頻度、文字列の長さ、中心性などを考慮して、統一後の代表名を決定する。
---

## 📂 ディレクトリ構成

ingredient-normalization/
├ src/
│ └ ingredient_norm/
│ ├ delta.py # 統一処理本体
│ └ param_search.py # パラメータ探索
├ scripts/
│ ├ run_delta.ps1
│ └ run_param.ps1
├ datasets/ # 入出力データ（gitignore）
├ README.md
└ .gitignore


---

## ⚙️ 使用技術

- Python 3.x
- sentence-transformers
- NetworkX
- NumPy / pandas
- SudachiPy

---

## 📥 入力データ

本プロジェクトでは、事前に前処理された材料データを使用します。

`datasets/` フォルダに以下のファイルを配置してください。

### labels_{name}.pickle

材料名と付与ラベルをまとめた前処理済みデータです。

中身例：

- name   : 材料名（文字列）
- labels : 抽出された特徴語・属性ラベル（list）

このファイルは、レシピデータからNER・形態素解析などにより
事前生成されます。

---

### scores_{name}.pickle

delta.py により計算された材料ペア間の類似度スコアです。

含まれる主な項目：

- term1 / term2 : 材料名ペア
- ce_score      : CrossEncoder類似度
- cos_score     : Sentence-BERT類似度
- string_score  : 文字列類似度（Jaro-Winkler）
- core_score    : 読み仮名特徴スコア
- total_score   : 融合後スコア

本ファイルは、param_search.py により
パラメータ最適化の入力として使用されます。




---

## ▶️ 実行方法

### ① 材料名統一（delta）

```powershell
.\scripts\run_delta.ps1

または：

python src/ingredient_norm/delta.py


実行後：

datasets/synonym_ners{name}.csv


が生成されます。

② パラメータ探索（param_search）
.\scripts\run_param.ps1


または：

python src/ingredient_norm/param_search.py


最適な α, β, γ, threshold が表示されます。
