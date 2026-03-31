# plateau-space-accessibility-viewer
PLATEAUの3D都市モデル（建物・土地利用データ）を用いて、任意エリアの空間指標（補正係数α）を算出・可視化するStreamlitアプリです。

🔗 **デモ**: [[Streamlit Cloud URL](https://plateau-space-accessibility-viewer.streamlit.app/)]

## 概要

補正係数αは、観光客が利用できる空間面積に対する地域関係者が利用できる空間面積の比率です。αが大きいほど地域利用の割合が高く、観光流入時の空間的負荷が大きいエリアといえます。

詳細な定義・分析結果については以下を参照してください：
- 〔査読論文〕

## 機能

- 地図上で矩形を描くだけで任意のエリアを指定
- 最大3エリアのαを並べて比較
- 土地利用カテゴリの色分け表示（3D/2D切替）
- 建物クリックで延床面積・階数を表示

## データの準備

PLATEAUのデータは[G空間情報センター](https://www.geospatial.jp/ckan/dataset/plateau)からダウンロードできます。変換にはQGISのPLATEAUプラグインまたはPLATEAU GIS Converterを使用してください。

```
data/
├── bldg/        # 建物GeoJSON（*.geojson）
├── luse/        # 土地利用GeoPackage（*.gpkg）
├── landuse_name_map.csv
├── landuse_category_roots.csv
└── landuse_category_groups.csv
```

## セットアップ

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 対応地域

PLATEAUがLOD1を整備している地域であれば原理的に適用可能です。デモ環境には東京都東京駅周辺のデータを収録しています。