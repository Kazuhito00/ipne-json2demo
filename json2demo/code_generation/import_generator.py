"""
Import文およびヘルパー関数生成モジュール

生成コードのimport文とヘルパー関数を生成
"""

from json2demo.helper_code.import_templates import MEDIAPIPE_IMPORT, ONNX_IMPORT
from json2demo.helper_code.mediapipe_helpers import DL_HELPERS
from json2demo.helper_code.onnx_helpers import ONNX_HELPERS, ONNX_HELPER_WHOLEBODY34
from json2demo.helper_code.onnx_yolox import ONNX_HELPER_YOLOX
from json2demo.helper_code.onnx_deimv2 import ONNX_HELPER_DEIMV2
from json2demo.helper_code.omnidirectional import OMNI_HELPER
from json2demo.helper_code.mot_helper import MOT_HELPER
from json2demo.helper_code.perf_helper import PERF_HELPER
from json2demo.imagenet_classes import IMAGENET_CLASS_NAMES


def generate_imports(
    needs_onnx,
    requires_numpy,
    enable_perf,
    used_node_types,
    needs_args=False,
):
    """
    import文を生成

    Returns:
        list: import文のコード行リスト
    """
    code_lines = []
    code_lines.append("#!/usr/bin/env python")
    code_lines.append("# -*- coding: utf-8 -*-")
    code_lines.append('"""')
    code_lines.append("Auto-generated OpenCV demo code from Node Editor JSON")
    code_lines.append("Press ESC or Q to exit")
    code_lines.append('"""')

    if needs_args:
        code_lines.append("import argparse")
    if needs_onnx:
        code_lines.append("import os")
    code_lines.append("import cv2")

    if requires_numpy or needs_onnx:
        code_lines.append("import numpy as np")
    if enable_perf:
        code_lines.append("import time")
    if "MOT" in used_node_types:
        code_lines.append("from motpy import Detection, MultiObjectTracker")
    if "ScreenCapture" in used_node_types:
        code_lines.append("from PIL import ImageGrab")

    return code_lines


def generate_dl_imports(needs_mediapipe, needs_onnx):
    """
    Deep Learning用のimport文を生成

    Returns:
        list: DLインポート文のコード行リスト
    """
    code_lines = []
    if needs_mediapipe:
        code_lines.append(MEDIAPIPE_IMPORT)
    if needs_onnx:
        code_lines.append(ONNX_IMPORT)
    return code_lines


def generate_helper_functions(
    used_node_types,
    face_uses_mediapipe,
    face_uses_facemesh,
    face_uses_yunet,
    od_model_types,
    enable_perf,
):
    """
    ヘルパー関数を生成

    Args:
        used_node_types: 使用されているノードタイプの集合
        face_uses_mediapipe: MediaPipe顔検出を使用するか
        face_uses_facemesh: FaceMeshを使用するか
        face_uses_yunet: YuNet顔検出を使用するか
        od_model_types: 使用するオブジェクト検出モデルタイプの集合
        enable_perf: パフォーマンス計測を有効にするか

    Returns:
        list: ヘルパー関数のコード行リスト
    """
    code_lines = []
    added_helpers = set()

    for node_type in used_node_types:
        if node_type == "FaceDetection":
            if face_uses_mediapipe and "FaceDetection_MediaPipe" not in added_helpers:
                code_lines.append(DL_HELPERS["FaceDetection_MediaPipe"])
                added_helpers.add("FaceDetection_MediaPipe")
            if face_uses_facemesh and "FaceDetection_FaceMesh" not in added_helpers:
                code_lines.append(DL_HELPERS["FaceDetection_FaceMesh"])
                added_helpers.add("FaceDetection_FaceMesh")
            if face_uses_yunet and "FaceDetection_YuNet" not in added_helpers:
                code_lines.append(ONNX_HELPERS["FaceDetection_YuNet"])
                added_helpers.add("FaceDetection_YuNet")
            continue

        if node_type == "ObjectDetection":
            if "wholebody34" in od_model_types:
                code_lines.append(ONNX_HELPER_WHOLEBODY34)
            if "deimv2" in od_model_types:
                code_lines.append(ONNX_HELPER_DEIMV2)
            if "yolox" in od_model_types:
                code_lines.append(ONNX_HELPER_YOLOX)
            added_helpers.add("ObjectDetection")
            continue

        if node_type in DL_HELPERS and node_type not in added_helpers:
            code_lines.append(DL_HELPERS[node_type])
            added_helpers.add(node_type)

        if node_type in ONNX_HELPERS and node_type not in added_helpers:
            helper_code = ONNX_HELPERS[node_type]
            if node_type == "Classification":
                helper_code = helper_code.replace(
                    "{imagenet_class_names}", repr(IMAGENET_CLASS_NAMES)
                )
            code_lines.append(helper_code)
            added_helpers.add(node_type)

    if "OmnidirectionalViewer" in used_node_types:
        code_lines.append(OMNI_HELPER)

    if "MOT" in used_node_types:
        code_lines.append(MOT_HELPER)

    if enable_perf:
        code_lines.append(PERF_HELPER)

    return code_lines


def generate_perf_helper():
    """
    パフォーマンス計測用ヘルパー関数のみを生成

    Returns:
        list: PERF_HELPERのコード行リスト
    """
    return [PERF_HELPER]
