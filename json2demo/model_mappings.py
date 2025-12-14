"""
モデル名とファイル名のマッピング、ノードタイプ定義
"""

# ONNXモデル名からファイル名へのマッピング
ONNX_MODEL_FILE_MAPPING = {
    # ObjectDetection - YOLOX
    "YOLOX-Nano(416x416)": "yolox_nano.onnx",
    "YOLOX-Tiny(416x416)": "yolox_tiny.onnx",
    "YOLOX-S(640x640)": "yolox_s.onnx",
    "FreeYOLO-Nano(640x640)": "yolo_free_nano_640x640.onnx",
    "FreeYOLO-Nano-CrowdHuman(640x640)": "yolo_free_nano_crowdhuman_640x640.onnx",
    "Light-Weight Person Detector": "light_weight_person_detector.onnx",
    # ObjectDetection - DEIMv2 (COCO)
    "DEIMv2-Atto(COCO)": "deimv2_hgnetv2_atto_coco.onnx",
    "DEIMv2-Femto(COCO)": "deimv2_hgnetv2_femto_coco.onnx",
    "DEIMv2-Pico(COCO)": "deimv2_hgnetv2_pico_coco.onnx",
    "DEIMv2-N(COCO)": "deimv2_hgnetv2_n_coco.onnx",
    "DEIMv2-S(COCO)": "deimv2_dinov3_s_coco.onnx",
    # ObjectDetection - DEIMv2 Wholebody34
    "DEIMv2-Wholebody34-Atto(320x320)": "deimv2_hgnetv2_atto_wholebody34_340query_n_batch_320x320.onnx",
    "DEIMv2-Wholebody34-Femto(416x416)": "deimv2_hgnetv2_femto_wholebody34_340query_n_batch_416x416.onnx",
    "DEIMv2-Wholebody34-Pico(640x640)": "deimv2_hgnetv2_pico_wholebody34_340query_n_batch_640x640.onnx",
    "DEIMv2-Wholebody34-N(640x640)": "deimv2_hgnetv2_n_wholebody34_680query_n_batch_640x640.onnx",
    "DEIMv2-Wholebody34-S(640x640)": "deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx",
    # Classification
    "MobileNetV3 Small": "MobileNetV3Small.onnx",
    "MobileNetV3 Large": "MobileNetV3Large.onnx",
    "EfficientNet B0": "EfficientNetB0.onnx",
    # MonocularDepthEstimation
    "FSRE-Depth(320x192)": "fsre_depth_192x320.onnx",
    "FSRE-Depth(640x384)": "fsre_depth_384x640.onnx",
    "Lite-HR-Depth(1280x384)": "lite_hr_depth_384x1280.onnx",
    "HR-Depth(1280x384)": "hr_depth_384x1280.onnx",
    # LLIE
    "TBEFN(320x180)": "tbefn_180x320.onnx",
    "TBEFN(640x360)": "tbefn_360x640.onnx",
    "SCI(320x180)": "sci_180x320.onnx",
    "SCI(640x360)": "sci_360x640.onnx",
    "AGLLNet(256x256)": "agllnet_256x256.onnx",
    "AGLLNet(512x384)": "agllnet_384x512.onnx",
    # FaceDetection
    "YuNet": "face_detection_yunet_120x160.onnx",
    # SemanticSegmentation
    "DeepLabV3": "deeplab_v3_1_default_1.onnx",
    "Road Segmentation ADAS 0001": "road_segmentation_adas_0001.onnx",
    "Skin Clothes Hair Segmentation": "skin_clothes_hair_segmentation.onnx",
    # PoseEstimation - MoveNet
    "MoveNet(SinglePose Lightning)": "movenet_singlepose_lightning_4.onnx",
    "MoveNet(SinglePose Thunder)": "movenet_singlepose_thunder_4.onnx",
    "MoveNet(MultiPose Lightning)": "movenet_multipose_lightning_1.onnx",
}

# PoseEstimationのモデル名によってHands/Pose/MoveNetを切り替え
POSE_MODEL_MAPPING = {
    "MediaPipe Hands(Complexity0)": "hands",
    "MediaPipe Hands(Complexity1)": "hands",
    "MediaPipe Pose(Complexity0)": "pose",
    "MediaPipe Pose(Complexity1)": "pose",
    "MediaPipe Pose(Complexity2)": "pose",
    "MoveNet(SinglePose Lightning)": "movenet",
    "MoveNet(SinglePose Thunder)": "movenet",
    "MoveNet(MultiPose Lightning)": "movenet",
}

# SemanticSegmentationのモデル名によってMediaPipe/ONNX(DeepLab等)を切り替え
SEGMENTATION_MODEL_MAPPING = {
    "MediaPipe SelfieSegmentation(Normal)": "mediapipe",
    "MediaPipe SelfieSegmentation(LandScape)": "mediapipe",
    "DeepLabV3": "deeplab",
    "Road Segmentation ADAS 0001": "deeplab",
    "Skin Clothes Hair Segmentation": "deeplab",
}

# MediaPipeモデル名のパターン
MEDIAPIPE_MODEL_PATTERNS = ["MediaPipe"]

# ONNXを使用するノードタイプ
ONNX_NODE_TYPES = {
    "ObjectDetection",
    "Classification",
    "MonocularDepthEstimation",
    "LLIE",
}

# FaceDetectionでONNXを使用するモデル名
ONNX_FACE_MODELS = {"YuNet"}

# numpyを使用するノードタイプ
NUMPY_NODES = {
    "GammaCorrection",
    "Sepia",
    "HSV",
    "SimpleFilter",
    "HistogramVisualization",
    "OmnidirectionalViewer",
    "ScreenCapture",
}

# numpyを使用するDLノードタイプ
NUMPY_DL_NODES = {
    "SemanticSegmentation",
    "ObjectDetection",
    "Classification",
    "MonocularDepthEstimation",
    "LLIE",
}
