"""
json2demo - Node Editor JSONからOpenCV HighGUIデモコードを生成するパッケージ

Usage:
    from json2demo import generate_code, parse_json

    # JSONファイルを解析
    data = parse_json("config.json")

    # コードを生成
    code = generate_code(data, output_path="output.py")
"""

from json2demo.json_parsing import (
    parse_json,
    extract_node_info,
    extract_links,
    topological_sort,
    get_input_vars,
    extract_setting_value,
    get_linked_param_value,
)

# 定数モジュールをエクスポート
from json2demo.node_param_mapping import NODE_PARAM_MAPPING
from json2demo.model_mappings import (
    ONNX_MODEL_FILE_MAPPING,
    POSE_MODEL_MAPPING,
    SEGMENTATION_MODEL_MAPPING,
    MEDIAPIPE_MODEL_PATTERNS,
    ONNX_NODE_TYPES,
    ONNX_FACE_MODELS,
    NUMPY_NODES,
    NUMPY_DL_NODES,
)
from json2demo.imagenet_classes import IMAGENET_CLASS_NAMES
from json2demo.node_templates import (
    NODE_TEMPLATES,
    INPUT_NODE_TEMPLATES,
    PROCESSING_NODE_TEMPLATES,
    OUTPUT_NODE_TEMPLATES,
    DL_NODE_TEMPLATES,
)
from json2demo.code_generation import generate_code

__all__ = [
    # コード生成
    "generate_code",
    # JSON解析関数
    "parse_json",
    "extract_node_info",
    "extract_links",
    "topological_sort",
    "get_input_vars",
    "extract_setting_value",
    "get_linked_param_value",
    # 定数
    "NODE_PARAM_MAPPING",
    "ONNX_MODEL_FILE_MAPPING",
    "POSE_MODEL_MAPPING",
    "SEGMENTATION_MODEL_MAPPING",
    "MEDIAPIPE_MODEL_PATTERNS",
    "ONNX_NODE_TYPES",
    "ONNX_FACE_MODELS",
    "NUMPY_NODES",
    "NUMPY_DL_NODES",
    "IMAGENET_CLASS_NAMES",
    # ノードテンプレート
    "NODE_TEMPLATES",
    "INPUT_NODE_TEMPLATES",
    "PROCESSING_NODE_TEMPLATES",
    "OUTPUT_NODE_TEMPLATES",
    "DL_NODE_TEMPLATES",
]
