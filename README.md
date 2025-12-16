# ipne-json2demo
[Image-Processing-Node-Editor](https://github.com/Kazuhito00/Image-Processing-Node-Editor)でExportしたノード設定(JSONファイル)から、<br>
スタンドアロンで動作するOpenCV HighGUIデモコードを自動生成するツールです。

ノードエディターで作成した画像処理パイプラインを、<br>
配布可能なPythonスクリプトとして出力できます。

# Note
```
opencv-python
mediapipe      ※MediaPipeノード使用時に必要
onnx           ※ONNXモデル使用時に必要
onnxruntime    ※ONNXモデル使用時に必要
motpy          ※MOTノード使用時に必要
pillow         ※ScreenCaptureノード使用時に必要
```

# Installation
```bash
# リポジトリをクローン
git clone https://github.com/YOUR_USERNAME/ipne-json2demo

# ディレクトリに移動
cd ipne-json2demo

# パッケージをインストール
pip install -r requirements.txt
```

# Usage
## 基本的な使い方
```bash
python -m json2demo <json_file> [output_file] [--perf]
```

### オプション
* json_file<br>
Image-Processing-Node-EditorでExportしたJSONファイルのパス
* output_file<br>
出力するPythonファイルのパス（デフォルト: output.py）
* --perf<br>
処理時間計測コードを含めて生成

### 使用例
```bash
# output.pyに出力（デフォルト）
python -m json2demo sample.json

# ファイル名を指定して出力
python -m json2demo sample.json demo.py

# 処理時間計測付きで生成
python -m json2demo sample.json demo_perf.py --perf
```

### 生成されたコードの実行
生成されたデモコードは以下のようなコマンドライン引数をサポートします（入力ノードに応じて異なります）。

```bash
# WebCamノードを使用する場合
python demo.py --camera 0 --width 640 --height 480

# Videoノードを使用する場合
python demo.py --video input.mp4

# Imageノードを使用する場合
python demo.py --image input.jpg
```

# Supported Node
<details>
<summary>Input Node（入力ノード）</summary>

| ノード名 | 説明 |
|---------|------|
| WebCam | Webカメラからの映像入力 |
| Video | 動画ファイル(mp4, avi)の読み込み |
| VideoSetFramePos | 動画の指定フレーム位置を読み込み |
| Image | 静止画(bmp, jpg, png等)の読み込み |
| RTSPInput | ネットワークカメラのRTSP入力 |
| ScreenCapture | デスクトップ画面のキャプチャ |
| IntValue | 整数値の出力 |
| FloatValue | 浮動小数点値の出力 |

</details>

<details>
<summary>Process Node（処理ノード）</summary>

| ノード名 | 説明 |
|---------|------|
| Blur | 平滑化（ぼかし）処理 |
| Grayscale | グレースケール変換 |
| Canny | キャニー法によるエッジ検出 |
| Threshold | 2値化処理 |
| Brightness | 輝度調整 |
| Contrast | コントラスト調整 |
| Flip | 水平/垂直反転 |
| Resize | リサイズ処理 |
| Crop | 切り抜き処理 |
| GammaCorrection | ガンマ補正 |
| EqualizeHist | ヒストグラム平坦化 |
| Sepia | セピア調変換 |
| ApplyColorMap | 疑似カラー適用 |
| HSV | HSV色空間での色調整 |
| SimpleFilter | 3x3フィルタ処理 |
| OmnidirectionalViewer | 360度画像のビューワー |

</details>

<details>
<summary>Deep Learning Node（深層学習ノード）</summary>

| ノード名 | 説明 |
|---------|------|
| FaceDetection | 顔検出（MediaPipe / YuNet） |
| PoseEstimation | 姿勢推定（MediaPipe Pose / MoveNet） |
| SemanticSegmentation | セマンティックセグメンテーション（MediaPipe / DeepLab） |
| HandDetection | 手検出（MediaPipe Hands） |
| ObjectDetection | 物体検出（YOLOX / DEIMv2 / Wholebody34） |
| Classification | 画像分類（MobileNetV3等） |
| MonocularDepthEstimation | 単眼深度推定（FSRE-Depth等） |
| LLIE | 暗所画像補正（TBEFN等） |

</details>

<details>
<summary>Analysis Node（分析ノード）</summary>

| ノード名 | 説明 |
|---------|------|
| RGBHistgram | RGBヒストグラム表示 |
| BRISQUE | 画質評価スコア表示 |

</details>

<details>
<summary>Draw Node（描画ノード）</summary>

| ノード名 | 説明 |
|---------|------|
| ImageConcat | 複数画像の結合表示 |
| ImageAlphaBlend | 2画像のアルファブレンド |
| PutText | テキスト描画 |
| ResultImage | 結果画像の表示 |
| ResultImageLarge | 結果画像の大きい表示 |

</details>

<details>
<summary>Other Node（その他ノード）</summary>

| ノード名 | 説明 |
|---------|------|
| VideoWriter | 動画ファイルへの書き出し |
| MOT | Multi Object Tracking |
| ExecPythonCode | カスタムPythonコード実行 |

</details>

# Example
sample.jsonを使用した生成例：

```bash
# デモコードを生成
python -m json2demo sample.json webcam_demo.py

# 生成されたコードを実行
python webcam_demo.py --camera 0
```

sample.jsonには以下のパイプラインが定義されています：
- WebCam入力
- Grayscale変換 → Resize
- PoseEstimation（姿勢推定）
- SemanticSegmentation（セグメンテーション）
- ImageConcat（4画像の結合表示）

# Author
Image-Processing-Node-Editor: 高橋かずひと(https://twitter.com/KzhtTkhs)

# License
ipne-json2demo is under [Apache-2.0 license](LICENSE).

本ツールはImage-Processing-Node-EditorのExportファイルを処理対象としています。<br>
生成されるコードで使用するモデルのライセンスは、各モデルのライセンスに従います。
