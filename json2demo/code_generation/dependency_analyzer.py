"""
依存関係分析モジュール

ノードグラフからMediaPipe/ONNX/numpy等の依存関係を分析
"""

from json2demo.model_mappings import (
    NUMPY_NODES,
    NUMPY_DL_NODES,
    MEDIAPIPE_MODEL_PATTERNS,
    ONNX_NODE_TYPES,
    ONNX_FACE_MODELS,
    POSE_MODEL_MAPPING,
    SEGMENTATION_MODEL_MAPPING,
)
from json2demo.node_templates import NODE_TEMPLATES
from json2demo.json_parsing import extract_setting_value


def check_requires_numpy(nodes):
    """numpyが必要かチェック"""
    for node_info in nodes.values():
        node_type = node_info["type"]
        if node_type in NUMPY_NODES or node_type in NUMPY_DL_NODES:
            return True
    return False


def check_requires_dl(nodes):
    """Deep Learningノードが含まれているかチェック"""
    dl_node_types = {
        "FaceDetection",
        "PoseEstimation",
        "SemanticSegmentation",
        "HandDetection",
        "ObjectDetection",
        "Classification",
        "MonocularDepthEstimation",
        "LLIE",
    }
    for node_info in nodes.values():
        node_type = node_info["type"]
        if node_type in dl_node_types:
            return True
        if node_type in NODE_TEMPLATES:
            template = NODE_TEMPLATES[node_type]
            if template.get("requires_dl", False):
                return True
    return False


def analyze_node_types(sorted_nodes, nodes, links):
    """
    ノードタイプの使用状況とMediaPipe/ONNX要否を分析

    Returns:
        tuple: (used_node_types, needs_mediapipe, needs_onnx,
                face_uses_yunet, face_uses_mediapipe, face_uses_facemesh,
                od_model_types)
    """
    used_node_types = set()
    needs_mediapipe = False
    needs_onnx = False
    face_uses_yunet = False
    face_uses_mediapipe = False
    face_uses_facemesh = False
    od_model_types = set()

    for node_key in sorted_nodes:
        node_info = nodes[node_key]
        node_type = node_info["type"]
        setting = node_info["setting"]

        if node_type == "PoseEstimation":
            model_name = extract_setting_value(
                setting,
                node_info["id"],
                node_type,
                "model",
                "MediaPipe Pose(Complexity0)",
            )
            model_category = POSE_MODEL_MAPPING.get(model_name, "pose")
            if model_category == "hands":
                used_node_types.add("HandDetection")
                needs_mediapipe = True
            elif model_category == "movenet":
                used_node_types.add("PoseEstimation_MoveNet")
                needs_onnx = True
            else:
                used_node_types.add("PoseEstimation")
                needs_mediapipe = True

        elif node_type == "FaceDetection":
            used_node_types.add(node_type)
            model_name = extract_setting_value(
                setting,
                node_info["id"],
                node_type,
                "model",
                "MediaPipe FaceDetection(~2m)",
            )
            if any(onnx_model in model_name for onnx_model in ONNX_FACE_MODELS):
                needs_onnx = True
                face_uses_yunet = True
            elif "FaceMesh" in model_name:
                needs_mediapipe = True
                face_uses_facemesh = True
            elif any(pattern in model_name for pattern in MEDIAPIPE_MODEL_PATTERNS):
                needs_mediapipe = True
                face_uses_mediapipe = True

        elif node_type == "SemanticSegmentation":
            model_name = extract_setting_value(
                setting,
                node_info["id"],
                node_type,
                "model",
                "MediaPipe SelfieSegmentation(Normal)",
            )
            model_category = SEGMENTATION_MODEL_MAPPING.get(model_name, "mediapipe")
            if model_category == "deeplab":
                used_node_types.add("SemanticSegmentation_DeepLab")
                needs_onnx = True
            else:
                used_node_types.add("SemanticSegmentation")
                needs_mediapipe = True

        elif node_type == "ObjectDetection":
            used_node_types.add(node_type)
            needs_onnx = True
            model_name = extract_setting_value(
                setting,
                node_info["id"],
                node_type,
                "model",
                "YOLOX-Nano(416x416)",
            )
            if "Wholebody34" in model_name:
                od_model_types.add("wholebody34")
            elif "DEIMv2" in model_name:
                od_model_types.add("deimv2")
            else:
                od_model_types.add("yolox")

        elif node_type in ONNX_NODE_TYPES:
            used_node_types.add(node_type)
            needs_onnx = True
        else:
            used_node_types.add(node_type)

    return (
        used_node_types,
        needs_mediapipe,
        needs_onnx,
        face_uses_yunet,
        face_uses_mediapipe,
        face_uses_facemesh,
        od_model_types,
    )


def get_actual_node_type(node_type, setting, node_info, nodes):
    """
    ノードタイプの実際の処理タイプを取得（モデル選択に応じた切り替え）

    Args:
        node_type: 元のノードタイプ
        setting: ノード設定
        node_info: ノード情報
        nodes: 全ノード情報

    Returns:
        str: 実際に使用するノードタイプ
    """
    if node_type == "PoseEstimation":
        model_name = extract_setting_value(
            setting,
            node_info["id"],
            node_type,
            "model",
            "MediaPipe Pose(Complexity0)",
        )
        model_category = POSE_MODEL_MAPPING.get(model_name, "pose")
        if model_category == "hands":
            return "HandDetection"
        elif model_category == "movenet":
            return "PoseEstimation_MoveNet"
        return "PoseEstimation"

    elif node_type == "FaceDetection":
        model_name = extract_setting_value(
            setting,
            node_info["id"],
            node_type,
            "model",
            "MediaPipe FaceDetection(~2m)",
        )
        if any(onnx_model in model_name for onnx_model in ONNX_FACE_MODELS):
            return "FaceDetection_YuNet"
        elif "FaceMesh" in model_name:
            return "FaceDetection_FaceMesh"
        return "FaceDetection_MediaPipe"

    elif node_type == "SemanticSegmentation":
        model_name = extract_setting_value(
            setting,
            node_info["id"],
            node_type,
            "model",
            "MediaPipe SelfieSegmentation(Normal)",
        )
        model_category = SEGMENTATION_MODEL_MAPPING.get(model_name, "mediapipe")
        if model_category == "deeplab":
            return "SemanticSegmentation_DeepLab"
        return "SemanticSegmentation"

    elif node_type == "ObjectDetection":
        model_name = extract_setting_value(
            setting,
            node_info["id"],
            node_type,
            "model",
            "YOLOX-Nano(416x416)",
        )
        if "Wholebody34" in model_name:
            return "ObjectDetection_Wholebody34"
        elif "DEIMv2" in model_name:
            return "ObjectDetection_DEIMv2"
        return "ObjectDetection"

    return node_type


def identify_skipped_nodes(sorted_nodes, nodes):
    """
    スキップされるノードを特定

    NODE_TEMPLATESに存在しない、またはoutput_type: "none"のノード

    Returns:
        set: スキップされるノードキーの集合
    """
    skipped_nodes = set()
    for node_key in sorted_nodes:
        node_info = nodes[node_key]
        node_type = node_info["type"]
        if node_type not in NODE_TEMPLATES:
            skipped_nodes.add(node_key)
        elif NODE_TEMPLATES[node_type].get("output_type") == "none":
            skipped_nodes.add(node_key)
    return skipped_nodes


def build_variable_mapping(sorted_nodes, nodes, links, skipped_nodes):
    """
    変数名マッピングを構築

    Args:
        sorted_nodes: トポロジカルソートされたノードリスト
        nodes: 全ノード情報
        links: リンク情報
        skipped_nodes: スキップされるノードの集合

    Returns:
        dict: node_key -> 変数名のマッピング
    """
    var_mapping = {}
    var_index = 0

    for node_key in sorted_nodes:
        if node_key in skipped_nodes:
            for link in links:
                if link["dst_node"] == node_key:
                    src_node = link["src_node"]
                    while src_node in skipped_nodes:
                        found = False
                        for link_inner in links:
                            if link_inner["dst_node"] == src_node:
                                src_node = link_inner["src_node"]
                                found = True
                                break
                        if not found:
                            break
                    var_mapping[node_key] = var_mapping.get(src_node, "frame")
                    break
            else:
                var_mapping[node_key] = "frame"
        else:
            var_mapping[node_key] = f"img_{var_index}"
            var_index += 1

    return var_mapping
