"""
生成されるPythonスクリプト用のimportテンプレート
"""

# Deep Learningノード用の追加import（分離）
MEDIAPIPE_IMPORT = """
# MediaPipe import
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not installed. Some features will be disabled.")"""

ONNX_IMPORT = """
# ONNX Runtime import
try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not installed. ONNX models will be disabled.")"""
