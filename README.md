# AI髪型シミュレーター 試作版

オープンキャンパス展示用の「髪型推薦・試着システム」の試作版です。
Webカメラで撮影した画像、またはアップロードした肩から上の人物画像に対して、背景を削除し、元画像の髪領域をFace Parsingで検出して透明化し、仮のボブヘアを顔位置に合わせて合成します。

今回は試作版のため、髪型は1種類のイラスト風ボブヘアのみです。顔型判定、髪型推薦、髪色変更、リアルタイム動画合成は実装していません。

## 使用技術

- Python
- Streamlit
- OpenCV
- MediaPipe Face Mesh
- rembg
- onnxruntime
- PyTorch
- BiSeNet Face Parsing
- Pillow
- NumPy

## ファイル構成

```text
hair-style-simulator/
├─ app.py
├─ image_processing.py
├─ face_detection.py
├─ face_parsing.py
├─ hair_overlay.py
├─ requirements.txt
├─ README.md
├─ assets/
│  ├─ bob_hair_back.png
│  └─ bob_hair_front.png
└─ models/
   └─ face_parsing_bisenet.pth
```

## 処理の流れ

```text
入力画像
↓
背景削除
↓
Face Parsingで元の髪領域を検出
↓
元の髪領域を透明化
↓
白背景
↓
仮ボブヘアの後ろ側レイヤー
↓
髪透明化済み人物
↓
仮ボブヘアの前側レイヤー
```

## Face Parsingとは

Face Parsingは、顔画像の各ピクセルを「肌」「眉」「目」「鼻」「口」「耳」「首」「服」「髪」などの意味ごとの領域に分類するセマンティックセグメンテーションです。

このアプリでは、CelebAMask-HQ用に学習されたBiSeNetの19クラスFace Parsingモデルを想定しています。
一般的なCelebAMask-HQ定義では、クラス17が髪領域です。

## モデルファイル

Face Parsingには、CelebAMask-HQ用BiSeNetのPyTorch重みファイルが必要です。
以下に配置してください。

```text
models/face_parsing_bisenet.pth
```

## 仮想環境の作成方法

PowerShellでこのフォルダへ移動してから、次のコマンドを実行します。
Pythonは3.10または3.11を推奨します。

```powershell
python -m venv .venv
```

仮想環境を有効化します。

```powershell
.\.venv\Scripts\Activate.ps1
```

## ライブラリのインストール方法

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 起動方法

```powershell
streamlit run app.py
```

## 操作方法

1. 「カメラで撮影」または「画像をアップロード」を選びます。
2. 正面に近い、肩から上が写った画像を入力します。
3. 背景削除、顔検出、髪領域検出が終わると、元の髪が透明化されます。
4. 白背景、後ろ髪レイヤー、髪透明化済み人物、前髪レイヤーの順に合成されます。
5. 必要に応じて「髪型サイズ」「左右位置」「上下位置」「回転角度」のスライダーで微調整します。
6. 「完成画像をPNGでダウンロード」ボタンで保存します。

## デバッグ表示

「デバッグ表示」にチェックを入れると、以下を確認できます。

- 入力画像
- 背景削除後の人物画像
- Face Parsingのクラス分類画像
- 髪領域マスク
- 髪領域の赤色プレビュー
- 元の髪を透明化した人物画像
- 推薦髪型の後ろ側レイヤー
- 推薦髪型の前側レイヤー
- 最終合成結果

## よくあるエラーと対処方法

### Face Parsingモデルが見つからない

`models/face_parsing_bisenet.pth` が存在するか確認してください。

### 髪領域を正しく検出できない

- 正面に近い写真を使ってください。
- 頭頂部まで写っている画像を使ってください。
- 暗い場所、帽子、手で髪が隠れている写真は避けてください。
- 「デバッグ表示」で髪マスクを確認してください。

### 顔が検出できない

- 正面に近い写真を使ってください。
- 顔が暗い、横向き、マスクや手で隠れている場合は検出しにくくなります。
- 顔が小さすぎる場合は、肩から上が大きく写った画像を使ってください。

### 背景削除に失敗する

- `rembg` と `onnxruntime` が正しくインストールされているか確認してください。
- 初回起動時はモデルのダウンロードに時間がかかることがあります。

## 重要な注意

元画像の髪を透明化すると、髪に隠れていた額、耳、頬、首、背景などは復元されません。
この試作版では、透明化した髪領域を仮のボブヘアで覆うことで自然に見えるようにしています。
