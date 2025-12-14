"""
出力コード生成モジュール

出力フェーズ（ResultImage等）と終端ノードの自動表示を生成
"""

import re

from json2demo.node_templates import NODE_TEMPLATES
from json2demo.json_parsing import get_input_vars
from json2demo.code_generation.utilities import get_ancestors


def _make_window_name(base_name, window_prefix):
    """ウィンドウ名を生成（プレフィックス付き）"""
    if window_prefix:
        return f"{window_prefix}:{base_name}"
    return base_name


def generate_output_phase(output_nodes, process_nodes, links, nodes, var_mapping, enable_perf, window_prefix=""):
    """
    出力フェーズ（ResultImage等）のコードを生成

    Returns:
        list: コード行リスト
    """
    code_lines = []

    if not output_nodes:
        return code_lines

    for node_data in output_nodes:
        node_key = node_data["node_key"]
        node_type = node_data["node_type"]
        actual_type = node_data["actual_type"]
        node_info = node_data["node_info"]

        template = NODE_TEMPLATES[actual_type]
        if not template["process"]:
            continue

        inputs = get_input_vars(node_key, links, var_mapping)
        input_var = inputs[0]["var"] if inputs else "frame"

        if enable_perf:
            ancestors = get_ancestors(node_key, links)
            for link in links:
                if link["dst_node"] == node_key:
                    ancestors.add(link["src_node"])

            perf_keys = []
            for pnode_data in process_nodes:
                if pnode_data["node_key"] in ancestors:
                    ntype = pnode_data["node_type"]
                    nid = nodes[pnode_data["node_key"]]["id"]
                    perf_keys.append(f"{ntype}_{nid}")

            node_id = node_info["id"]
            code_lines.append(f"        # {node_type} (with perf info)")
            code_lines.append(
                "        _perf_total = (time.perf_counter() - _perf_total_start) * 1000"
            )
            code_lines.append(
                "        _perf_fps = 1000.0 / _perf_total if _perf_total > 0 else 0"
            )
            perf_keys_str = repr(perf_keys)
            code_lines.append(
                f"        _perf_img_{node_id} = draw_perf_info({input_var}, _perf_times, {perf_keys_str}, _perf_total, _perf_fps)"
            )

            if actual_type == "VideoWriter":
                code_lines.append(f"        if _video_writer{node_id} is None:")
                code_lines.append(f"            _h{node_id}, _w{node_id} = _perf_img_{node_id}.shape[:2]")
                code_lines.append(f'            _video_writer{node_id} = cv2.VideoWriter("output_{node_id}.mp4", _fourcc{node_id}, _cap_fps, (_w{node_id}, _h{node_id}))')
                code_lines.append(f"        _video_writer{node_id}.write(_perf_img_{node_id})")

            if actual_type == "ResultImage":
                base_name = f"Result_{node_id}"
            elif actual_type == "ResultImageLarge":
                base_name = f"ResultLarge_{node_id}"
            else:
                base_name = f"{node_type}_{node_id}"
            window_name = _make_window_name(base_name, window_prefix)

            code_lines.append(f'        cv2.imshow("{window_name}", _perf_img_{node_id})')
            code_lines.append("")
        else:
            process_code = template["process"]
            process_code = process_code.replace("{input}", input_var)
            output_var = var_mapping[node_key]
            process_code = process_code.replace("{output}", output_var)
            process_code = process_code.replace("{node_id}", node_info["id"])
            process_code = process_code.replace("{window_prefix}", window_prefix)

            # imshow内のウィンドウ名にプレフィックスを追加
            if window_prefix and 'cv2.imshow("' in process_code:
                def add_prefix(match):
                    return f'cv2.imshow("{window_prefix}:{match.group(1)}"'
                process_code = re.sub(r'cv2\.imshow\("([^"]+)"', add_prefix, process_code)

            code_lines.append(f"        # {node_type}")
            for line in process_code.split("\n"):
                code_lines.append(f"        {line}")
            code_lines.append("")

    return code_lines


def generate_terminal_nodes_display(process_nodes, output_nodes, links, nodes, var_mapping, enable_perf, window_prefix=""):
    """
    終端ノード（出力がリンクされていない画像ノード）の自動表示コードを生成

    Returns:
        list: コード行リスト
    """
    code_lines = []

    src_nodes = set(link["src_node"] for link in links)
    output_input_nodes = set()
    for node_data in output_nodes:
        inputs = get_input_vars(node_data["node_key"], links, var_mapping)
        for link in links:
            if link["dst_node"] == node_data["node_key"]:
                output_input_nodes.add(link["src_node"])

    terminal_image_vars = []
    for node_data in process_nodes:
        actual_type = node_data["actual_type"]
        template = NODE_TEMPLATES[actual_type]
        if template["output_type"] == "image":
            node_key = node_data["node_key"]
            if node_key not in src_nodes and node_key not in output_input_nodes:
                terminal_image_vars.append({
                    "var": var_mapping[node_key],
                    "node_type": node_data["node_type"],
                    "node_id": nodes[node_key]["id"],
                    "node_key": node_key,
                })

    if not terminal_image_vars:
        return code_lines

    terminal_perf_keys = {}
    for term_info in terminal_image_vars:
        node_key = term_info["node_key"]
        ancestors = get_ancestors(node_key, links)
        ancestors.add(node_key)
        perf_keys = []
        for node_data in process_nodes:
            if node_data["node_key"] in ancestors:
                ntype = node_data["node_type"]
                nid = nodes[node_data["node_key"]]["id"]
                perf_keys.append(f"{ntype}_{nid}")
        terminal_perf_keys[node_key] = perf_keys

    code_lines.append("        # Auto-generated imshow (additional terminal nodes)")
    if enable_perf:
        code_lines.append(
            "        _perf_total = (time.perf_counter() - _perf_total_start) * 1000"
        )
        code_lines.append(
            "        _perf_fps = 1000.0 / _perf_total if _perf_total > 0 else 0"
        )
        code_lines.append("")

    for term_info in terminal_image_vars:
        base_name = f"{term_info['node_type']}_{term_info['node_id']}"
        window_name = _make_window_name(base_name, window_prefix)
        if enable_perf:
            perf_keys = terminal_perf_keys[term_info["node_key"]]
            perf_keys_str = repr(perf_keys)
            code_lines.append(
                f"        _perf_img_{term_info['node_id']} = draw_perf_info({term_info['var']}, _perf_times, {perf_keys_str}, _perf_total, _perf_fps)"
            )
            code_lines.append(
                f'        cv2.imshow("{window_name}", _perf_img_{term_info["node_id"]})'
            )
        else:
            code_lines.append(f'        cv2.imshow("{window_name}", {term_info["var"]})')

    code_lines.append("")
    return code_lines


def generate_key_handling():
    """
    キー入力処理コードを生成

    Returns:
        list: コード行リスト
    """
    code_lines = []
    code_lines.append("        # Exit on ESC or Q key")
    code_lines.append("        key = cv2.waitKey(1) & 0xFF")
    code_lines.append('        if key == 27 or key == ord("q"):')
    code_lines.append("            break")
    code_lines.append("")
    return code_lines


def generate_main_footer():
    """
    メイン関数のフッターを生成

    Returns:
        list: コード行リスト
    """
    code_lines = []
    code_lines.append("")
    code_lines.append('if __name__ == "__main__":')
    code_lines.append("    main()")
    code_lines.append("")
    return code_lines
