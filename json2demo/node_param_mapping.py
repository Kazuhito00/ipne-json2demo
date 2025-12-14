"""
ノードタイプごとのパラメータとJSONキーのマッピング定義
"""

NODE_PARAM_MAPPING = {
    # 画像処理ノード
    "Blur": {
        "kernel_size": "Int:Input02Value",
    },
    "Brightness": {
        "brightness": "Int:Input02Value",
    },
    "Contrast": {
        "contrast": "Float:Input02Value",
    },
    "GammaCorrection": {
        "gamma": "Float:Input02Value",
    },
    "Canny": {
        "canny_min": "Int:Input02Value",
        "canny_max": "Int:Input03Value",
    },
    "Threshold": {
        "threshold_type": "Text:Input02Value",
        "binary_threshold": "Int:Input03Value",
    },
    "HSV": {
        "h_add": "Int:Input02Value",
        "s_add": "Int:Input03Value",
        "v_add": "Int:Input04Value",
    },
    "ApplyColorMap": {
        "colormap": "Text:Input02Value",
    },
    "Flip": {
        "hflip": "Text:Input02Value",
        "vflip": "Text:Input03Value",
    },
    "Resize": {
        "width": "Int:Input02Value",
        "height": "Int:Input03Value",
        "interpolation": "Text:Input04Value",
    },
    "Crop": {
        "min_x": "Float:Input02Value",
        "max_x": "Float:Input03Value",
        "min_y": "Float:Input04Value",
        "max_y": "Float:Input05Value",
    },
    "SimpleFilter": {
        "x0y0": "Float:Input02Value",
        "x1y0": "Float:Input03Value",
        "x2y0": "Float:Input04Value",
        "x0y1": "Float:Input05Value",
        "x1y1": "Float:Input06Value",
        "x2y1": "Float:Input07Value",
        "x0y2": "Float:Input08Value",
        "x1y2": "Float:Input09Value",
        "x2y2": "Float:Input10Value",
        "filter_k": "Float:Input11Value",
    },
    # 描画ノード
    "PutText": {
        "text": "Text:Input02Value",
    },
    "ImageAlphaBlend": {
        "alpha": "Float:Input03Value",
        "beta": "Float:Input04Value",
        "gamma_blend": "Int:Input05Value",
    },
    # 入力ノード
    "Video": {
        "video_path": "Text:Input02Value",
        "skip_frame": "Int:Input03Value",
    },
    "Image": {
        "image_path": "Text:Input01Value",
    },
    "RTSPInput": {
        "rtsp_url": "Text:Input01Value",
    },
    "VideoSetFramePos": {
        "frame_pos": "Int:Input02Value",
    },
    "IntValue": {
        "int_value": "Int:Input01Value",
    },
    "FloatValue": {
        "float_value": "Float:Input01Value",
    },
    "ExecPythonCode": {
        "code": "Text:Input02Value",
    },
    "MOT": {
        "max_staleness": "Int:Input02Value",
    },
    "ScreenCapture": {
        "monitor": "Int:Input01Value",
    },
    # Deep Learningノード
    "FaceDetection": {
        "model": "Text:Input02Value",
        "score_threshold": "Float:Input03Value",
    },
    "FaceDetection_MediaPipe": {
        "model": "Text:Input02Value",
        "score_threshold": "Float:Input03Value",
    },
    "FaceDetection_YuNet": {
        "model": "Text:Input02Value",
        "score_threshold": "Float:Input03Value",
    },
    "PoseEstimation": {
        "model": "Text:Input02Value",
        "score_threshold": "Float:Input03Value",
    },
    "HandDetection": {
        "model": "Text:Input02Value",
        "score_threshold": "Float:Input03Value",
    },
    "SemanticSegmentation": {
        "model": "Text:Input02Value",
        "score_threshold": "Float:Input03Value",
    },
    "ObjectDetection": {
        "model": "Text:Input02Value",
        "score_threshold": "Float:Input03Value",
    },
    "Classification": {
        "model": "Text:Input02Value",
    },
    "MonocularDepthEstimation": {
        "model": "Text:Input02Value",
    },
    "LLIE": {
        "model": "Text:Input02Value",
    },
    # 全天球ビューワー
    "OmnidirectionalViewer": {
        "pitch": "Int:Input02Value",
        "yaw": "Int:Input03Value",
        "roll": "Int:Input04Value",
        "imagepoint": "Float:Input05Value",
    },
}
