# AI髪型シミュレーター 試作版

オープンキャンパス展示用の「髪型推薦・試着システム」の試作版です。
Webカメラで撮影した画像、またはアップロードした肩から上の人物画像に対して、背景を削除し、Face Parsingで元画像の髪領域を透明化し、`C:\Users\airic\Desktop\output` に保存された髪型PNG素材を合成します。

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
└─ models/
   └─ face_parsing_bisenet.pth
```

髪型素材はプロジェクト外の以下から自動で読み込みます。

```text
C:\Users\airic\Desktop\output
```

## 髪型素材の構成

`Desktop/output` の直下に髪型ごとのフォルダを置きます。
`metadata.json` が存在するフォルダだけを髪型素材として認識します。

```text
output/
├─ bob_1/
│  ├─ front.png
│  ├─ side_left.png
│  ├─ side_right.png
│  ├─ strands.png
│  ├─ hair_full.png
│  └─ metadata.json
├─ bob_2/
├─ short_1/
└─ long_1/
```

アプリを再起動すると、新しく追加した髪型フォルダもサイドバーの「髪型を選択」に自動で表示されます。

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
髪透明化済み人物
↓
side_left.png
↓
side_right.png
↓
front.png
↓
strands.png
```

## 髪型の位置合わせ

髪型素材の `metadata.json` に `anchors` がある場合、以下の3点を人物側の対応点へ合わせます。

```json
{
  "anchors": {
    "left_temple": [215, 260],
    "right_temple": [427, 260],
    "face_bottom": [320, 470],
    "bang_center": [320, 235],
    "hair_top": [320, 90]
  }
}
```

`anchors` がない場合は、metadata内の顔ランドマークと `hair_full.png` のアルファ領域から初期基準点を推定します。
変換には `cv2.estimateAffinePartial2D` と `cv2.warpAffine` を使い、`hair_full`、`side_left`、`side_right`、`front`、`strands` の全レイヤーへ同じアフィン変換を適用します。

## Face Parsingとは

Face Parsingは、顔画像の各ピクセルを「肌」「眉」「目」「鼻」「口」「耳」「首」「服」「髪」などの意味ごとの領域に分類するセマンティックセグメンテーションです。

このアプリでは、CelebAMask-HQ用に学習されたBiSeNetの19クラスFace Parsingモデルを想定しています。
一般的なCelebAMask-HQ定義では、クラス17が髪領域です。

## 起動方法

PowerShellで以下を実行します。

```powershell
cd C:\Users\airic\Desktop\hair-style-simulator
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## 操作方法

1. サイドバーの「髪型を選択」から髪型を選びます。
2. 「カメラで撮影」または「画像をアップロード」を選びます。
3. 正面に近い、肩から上が写った画像を入力します。
4. 背景削除、顔検出、髪領域検出が終わると、元の髪が透明化されます。
5. 選択した髪型素材が `side_left`、`side_right`、`front`、`strands` の順に合成されます。
6. 必要に応じて「髪型サイズ」「左右位置」「上下位置」「回転角度」のスライダーで微調整します。
7. 「完成画像をPNGでダウンロード」ボタンで保存します。

## デバッグ表示

「デバッグ表示」にチェックを入れると、以下を確認できます。

- 入力画像
- 背景削除後の人物画像
- Face Parsingのクラス分類画像
- 髪領域マスク
- 髪領域の赤色プレビュー
- 元の髪を透明化した人物画像
- `side_left.png`
- `side_right.png`
- `front.png`
- `strands.png`
- 最終合成結果

## よくあるエラー

### 髪型素材が表示されない

`C:\Users\airic\Desktop\output` の中に、`metadata.json` を持つ髪型フォルダがあるか確認してください。
各髪型フォルダには `front.png`、`side_left.png`、`side_right.png`、`strands.png` が必要です。

### Face Parsingモデルが見つからない

`models/face_parsing_bisenet.pth` が存在するか確認してください。

### 髪領域を正しく検出できない

- 正面に近い写真を使ってください。
- 頭頂部まで写っている画像を使ってください。
- 暗い場所、帽子、手で髪が隠れている写真は避けてください。
- 「デバッグ表示」で髪マスクを確認してください。
