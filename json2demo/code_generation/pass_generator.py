"""
パス処理コード生成モジュール

メインループ内の4パス処理を生成:
- パス1: drawがないノード（画像変換系）
- パス2: drawがあるノードの検出フェーズ
- パス3: drawがあるノードの描画フェーズ
- パス4: deferred_nodes（drawがあるノードの後に処理）
"""

import re

from json2demo.node_templates import NODE_TEMPLATES
from json2demo.model_mappings import ONNX_MODEL_FILE_MAPPING
from json2demo.json_parsing import (
    extract_setting_value,
    get_linked_param_value,
    get_input_vars,
)
from json2demo.code_generation.dependency_analyzer import get_actual_node_type
from json2demo.code_generation.utilities import generate_image_concat_code


def _add_imshow_prefix(code, window_prefix):
    """imshowのウィンドウ名にプレフィックスを追加"""
    if window_prefix and 'cv2.imshow("' in code:
        def add_prefix(match):
            return f'cv2.imshow("{window_prefix}:{match.group(1)}"'
        code = re.sub(r'cv2\.imshow\("([^"]+)"', add_prefix, code)
    return code


def classify_nodes(sorted_nodes, nodes, links, var_mapping):
    """
    ノードを分類（処理ノード、出力ノード、スキップノード）

    Returns:
        tuple: (process_nodes, output_nodes)
    """
    process_nodes = []
    output_nodes = []

    for node_key in sorted_nodes:
        node_info = nodes[node_key]
        node_type = node_info["type"]
        setting = node_info["setting"]

        actual_type = get_actual_node_type(node_type, setting, node_info, nodes)

        if actual_type not in NODE_TEMPLATES:
            continue

        template = NODE_TEMPLATES[actual_type]

        if template.get("skip_codegen", False):
            inputs = get_input_vars(node_key, links, var_mapping)
            if inputs:
                var_mapping[node_key] = inputs[0]["var"]
            process_nodes.append({
                "node_key": node_key,
                "node_type": node_type,
                "actual_type": actual_type,
                "setting": setting,
                "node_info": node_info,
                "skip_codegen": True,
            })
            continue

        if template["output_type"] == "display":
            output_nodes.append({
                "node_key": node_key,
                "node_type": node_type,
                "actual_type": actual_type,
                "setting": setting,
                "node_info": node_info,
            })
            continue

        process_nodes.append({
            "node_key": node_key,
            "node_type": node_type,
            "actual_type": actual_type,
            "setting": setting,
            "node_info": node_info,
        })

    return process_nodes, output_nodes


def build_detect_input_mapping(process_nodes, links, var_mapping):
    """
    検出フェーズ用の入力画像マッピングを作成

    Returns:
        dict: node_key -> 検出時に使用する入力変数名
    """
    detect_input_mapping = {}
    last_image_var = None

    for node_data in process_nodes:
        node_key = node_data["node_key"]
        actual_type = node_data["actual_type"]
        template = NODE_TEMPLATES[actual_type]

        if "draw" not in template:
            last_image_var = var_mapping[node_key]
        else:
            inputs = get_input_vars(node_key, links, var_mapping)
            if inputs:
                input_var = inputs[0]["var"]
                detect_input_mapping[node_key] = (
                    last_image_var if last_image_var else input_var
                )

    return detect_input_mapping


def has_draw_input(node_key, links, nodes, var_mapping, visited=None):
    """
    入力元にdrawを持つノードがあるかチェック（推移的）

    Args:
        node_key: チェックするノードキー
        links: リンク情報
        nodes: 全ノード情報
        var_mapping: 変数名マッピング
        visited: 訪問済みノード集合

    Returns:
        bool: drawを持つ入力元があればTrue
    """
    if visited is None:
        visited = set()
    if node_key in visited:
        return False
    visited.add(node_key)

    inputs = get_input_vars(node_key, links, var_mapping)
    for inp in inputs:
        for nk, nd in nodes.items():
            if var_mapping.get(nk) == inp["var"]:
                nt = nd["type"]
                if nt in NODE_TEMPLATES and "draw" in NODE_TEMPLATES[nt]:
                    return True
                if has_draw_input(nk, links, nodes, var_mapping, visited):
                    return True
    return False


def _inject_perf_drawing(process_code, input_var, node_id):
    """
    テンプレート内のimshow呼び出しにperf描画を追加

    入力画像を表示するimshowの前に、perf情報を描画するコードを挿入
    """
    import re
    lines = process_code.split('\n')
    result_lines = []
    perf_injected = False

    for line in lines:
        # 入力変数を表示するimshowを探す
        if not perf_injected and 'cv2.imshow(' in line and input_var in line:
            # perf計算とdraw_perf_infoを挿入
            result_lines.append(
                '_perf_total = (time.perf_counter() - _perf_total_start) * 1000'
            )
            result_lines.append(
                '_perf_fps = 1000.0 / _perf_total if _perf_total > 0 else 0'
            )
            result_lines.append(
                f'_perf_img_{node_id} = draw_perf_info({input_var}, _perf_times, list(_perf_times.keys()), _perf_total, _perf_fps)'
            )
            # imshow内の変数をperf描画済み画像に置換
            line = line.replace(input_var, f'_perf_img_{node_id}')
            perf_injected = True
        result_lines.append(line)

    return '\n'.join(result_lines)


def _generate_process_code(node_data, links, nodes, var_mapping, enable_perf, window_prefix=""):
    """
    単一ノードの処理コードを生成

    Returns:
        list: コード行リスト
    """
    code_lines = []
    node_key = node_data["node_key"]
    node_type = node_data["node_type"]
    actual_type = node_data["actual_type"]
    setting = node_data["setting"]
    node_info = node_data["node_info"]

    template = NODE_TEMPLATES[actual_type]
    inputs = get_input_vars(node_key, links, var_mapping)
    output_var = var_mapping[node_key]

    if template.get("dynamic_codegen"):
        if actual_type == "ImageConcat":
            process_code = generate_image_concat_code(inputs, output_var)
        else:
            return code_lines
    elif not template["process"]:
        return code_lines
    else:
        process_code = template["process"]

        if inputs:
            process_code = process_code.replace("{input}", inputs[0]["var"])
            if len(inputs) > 1:
                process_code = process_code.replace("{input1}", inputs[0]["var"])
                process_code = process_code.replace("{input2}", inputs[1]["var"])
        else:
            process_code = process_code.replace("{input}", "frame")

        process_code = process_code.replace("{output}", output_var)

        param_values = {}
        for param, default in template["params"].items():
            linked_value = get_linked_param_value(node_key, param, links, nodes)
            if linked_value is not None:
                value = linked_value
            else:
                value = extract_setting_value(
                    setting, node_info["id"], node_info["type"], param, default
                )
            param_values[param] = value
            if isinstance(value, str):
                process_code = process_code.replace(f"{{{param}}}", value)
            else:
                process_code = process_code.replace(f"{{{param}}}", str(value))

        if node_type == "Flip":
            hflip = param_values.get("hflip", False)
            vflip = param_values.get("vflip", False)
            if hflip and vflip:
                flipcode = -1
            elif hflip:
                flipcode = 1
            else:
                flipcode = 0
            process_code = process_code.replace("{flipcode}", str(flipcode))

        process_code = process_code.replace("{node_id}", node_info["id"])

    # imshowのウィンドウ名にプレフィックスを追加
    process_code = _add_imshow_prefix(process_code, window_prefix)

    # テンプレート内にimshowがある場合、perf描画を追加
    if enable_perf and 'cv2.imshow(' in process_code and inputs:
        input_var = inputs[0]["var"]
        process_code = _inject_perf_drawing(process_code, input_var, node_info["id"])

    node_id = node_info["id"]
    code_lines.append(f"        # {node_type}")
    if enable_perf:
        code_lines.append("        _perf_start = time.perf_counter()")
    for line in process_code.split("\n"):
        code_lines.append(f"        {line}")
    if enable_perf:
        code_lines.append(
            f"        _perf_times['{node_type}_{node_id}'] = (time.perf_counter() - _perf_start) * 1000"
        )
    code_lines.append("")

    return code_lines


def generate_pass1_transformations(process_nodes, links, nodes, var_mapping, enable_perf, window_prefix=""):
    """
    パス1: drawがないノードを処理（入力元にdrawがあるノードは除外）

    Returns:
        tuple: (code_lines, deferred_nodes)
    """
    code_lines = []
    deferred_nodes = []

    for node_data in process_nodes:
        if node_data.get("skip_codegen", False):
            continue

        node_key = node_data["node_key"]
        actual_type = node_data["actual_type"]
        template = NODE_TEMPLATES[actual_type]

        if "draw" in template:
            continue

        if has_draw_input(node_key, links, nodes, var_mapping):
            deferred_nodes.append(node_data)
            continue

        code_lines.extend(_generate_process_code(node_data, links, nodes, var_mapping, enable_perf, window_prefix))

    return code_lines, deferred_nodes


def generate_pass2_detection(process_nodes, links, nodes, var_mapping,
                             detect_input_mapping, enable_perf, window_prefix=""):
    """
    パス2: drawがあるノードの検出フェーズ

    Returns:
        list: コード行リスト
    """
    code_lines = []

    for node_data in process_nodes:
        if node_data.get("skip_codegen", False):
            continue

        node_key = node_data["node_key"]
        node_type = node_data["node_type"]
        actual_type = node_data["actual_type"]
        setting = node_data["setting"]
        node_info = node_data["node_info"]

        template = NODE_TEMPLATES[actual_type]

        if "draw" not in template:
            continue

        if not template["process"]:
            continue

        process_code = template["process"]

        detect_input = detect_input_mapping.get(node_key)
        if detect_input:
            process_code = process_code.replace("{input}", detect_input)
        else:
            inputs = get_input_vars(node_key, links, var_mapping)
            if inputs:
                process_code = process_code.replace("{input}", inputs[0]["var"])
            else:
                process_code = process_code.replace("{input}", "frame")

        output_var = var_mapping[node_key]
        process_code = process_code.replace("{output}", output_var)

        for param, default in template["params"].items():
            linked_value = get_linked_param_value(node_key, param, links, nodes)
            if linked_value is not None:
                value = linked_value
            else:
                value = extract_setting_value(
                    setting, node_info["id"], node_info["type"], param, default
                )
            if isinstance(value, str):
                process_code = process_code.replace(f"{{{param}}}", value)
            else:
                process_code = process_code.replace(f"{{{param}}}", str(value))

        process_code = process_code.replace("{node_id}", node_info["id"])

        if template.get("requires_detection"):
            src_boxes = "[]"
            for link in links:
                if link["dst_node"] == node_key:
                    src_node = link["src_node"]
                    if src_node in nodes:
                        src_type = nodes[src_node]["type"]
                        src_id = nodes[src_node]["id"]
                        if src_type == "ObjectDetection":
                            src_boxes = f"_od{src_id}_boxes"
                            break
            process_code = process_code.replace("{src_boxes}", src_boxes)

        # imshowのウィンドウ名にプレフィックスを追加
        process_code = _add_imshow_prefix(process_code, window_prefix)

        node_id = node_info["id"]
        code_lines.append(f"        # {node_type}")
        if enable_perf:
            code_lines.append("        _perf_start = time.perf_counter()")
        for line in process_code.split("\n"):
            code_lines.append(f"        {line}")
        if enable_perf:
            code_lines.append(
                f"        _perf_times['{node_type}_{node_id}'] = (time.perf_counter() - _perf_start) * 1000"
            )
        code_lines.append("")

    return code_lines


def generate_pass3_drawing(process_nodes, links, nodes, var_mapping, enable_perf, window_prefix=""):
    """
    パス3: drawがあるノードの描画フェーズ

    Returns:
        list: コード行リスト
    """
    code_lines = []

    skip_draw_nodes = set()
    for node_data in process_nodes:
        node_key = node_data["node_key"]
        actual_type = node_data["actual_type"]
        template = NODE_TEMPLATES.get(actual_type, {})
        if template.get("skip_src_draw"):
            for link in links:
                if link["dst_node"] == node_key:
                    skip_draw_nodes.add(link["src_node"])

    for node_data in process_nodes:
        if node_data.get("skip_codegen", False):
            continue

        node_key = node_data["node_key"]
        node_type = node_data["node_type"]
        actual_type = node_data["actual_type"]
        setting = node_data["setting"]
        node_info = node_data["node_info"]

        template = NODE_TEMPLATES[actual_type]

        if "draw" not in template:
            continue

        if node_key in skip_draw_nodes:
            continue

        draw_code = template["draw"]

        inputs = get_input_vars(node_key, links, var_mapping)
        output_var = var_mapping[node_key]

        if inputs:
            draw_code = draw_code.replace("{input}", inputs[0]["var"])
            if len(inputs) > 1:
                draw_code = draw_code.replace("{input1}", inputs[0]["var"])
                draw_code = draw_code.replace("{input2}", inputs[1]["var"])
        else:
            draw_code = draw_code.replace("{input}", "frame")

        draw_code = draw_code.replace("{output}", output_var)

        for param, default in template["params"].items():
            linked_value = get_linked_param_value(node_key, param, links, nodes)
            if linked_value is not None:
                value = linked_value
            else:
                value = extract_setting_value(
                    setting, node_info["id"], node_info["type"], param, default
                )
            if isinstance(value, str):
                draw_code = draw_code.replace(f"{{{param}}}", value)
            else:
                draw_code = draw_code.replace(f"{{{param}}}", str(value))

        draw_code = draw_code.replace("{node_id}", node_info["id"])

        if template.get("requires_detection"):
            src_input = "frame"
            for link in links:
                if link["dst_node"] == node_key:
                    src_node = link["src_node"]
                    if src_node in nodes:
                        for link2 in links:
                            if link2["dst_node"] == src_node:
                                src_input = var_mapping.get(link2["src_node"], "frame")
                                break
                        break
            draw_code = draw_code.replace("{src_input}", src_input)

        # imshowのウィンドウ名にプレフィックスを追加
        draw_code = _add_imshow_prefix(draw_code, window_prefix)

        node_id = node_info["id"]
        code_lines.append(f"        # {node_type} (draw)")
        if enable_perf:
            code_lines.append("        _perf_start = time.perf_counter()")
        for line in draw_code.split("\n"):
            code_lines.append(f"        {line}")
        if enable_perf:
            code_lines.append(
                f"        _perf_times['{node_type}_{node_id}_draw'] = (time.perf_counter() - _perf_start) * 1000"
            )
        code_lines.append("")

    return code_lines


def generate_pass4_deferred(deferred_nodes, links, nodes, var_mapping, enable_perf, window_prefix=""):
    """
    パス4: drawがあるノードの後に処理するノード

    Returns:
        list: コード行リスト
    """
    code_lines = []

    for node_data in deferred_nodes:
        code_lines.extend(_generate_process_code(node_data, links, nodes, var_mapping, enable_perf, window_prefix))

    return code_lines
