"""
コード生成ユーティリティ関数

ImageConcat等の動的コード生成や共通処理
"""


def generate_image_concat_code(inputs, output_var, base_cell_width=640, base_cell_height=360):
    """
    ImageConcat用のコードを動的生成（アスペクト比16:9を維持）

    Args:
        inputs: 入力ノードリスト
        output_var: 出力変数名
        base_cell_width: セル幅
        base_cell_height: セル高さ

    Returns:
        str: 生成されたコード
    """
    input_count = len(inputs)
    input_vars = [inp["var"] for inp in reversed(inputs)]
    lines = []

    cell_width, cell_height = base_cell_width, base_cell_height

    if input_count == 1:
        lines.append(f"{output_var} = cv2.resize({input_vars[0]}, ({cell_width}, {cell_height}))")

    elif input_count == 2:
        for index, var in enumerate(input_vars):
            lines.append(f"_c{index} = cv2.resize({var}, ({cell_width}, {cell_height}))")
        lines.append(f"{output_var} = cv2.hconcat([_c0, _c1])")

    elif input_count == 3:
        for index, var in enumerate(input_vars):
            lines.append(f"_c{index} = cv2.resize({var}, ({cell_width}, {cell_height}))")
        lines.append(f"_c3 = np.zeros(({cell_height}, {cell_width}, 3), dtype=np.uint8)")
        lines.append("_row0 = cv2.hconcat([_c0, _c1])")
        lines.append("_row1 = cv2.hconcat([_c2, _c3])")
        lines.append(f"{output_var} = cv2.vconcat([_row0, _row1])")

    elif input_count == 4:
        for index, var in enumerate(input_vars):
            lines.append(f"_c{index} = cv2.resize({var}, ({cell_width}, {cell_height}))")
        lines.append("_row0 = cv2.hconcat([_c0, _c1])")
        lines.append("_row1 = cv2.hconcat([_c2, _c3])")
        lines.append(f"{output_var} = cv2.vconcat([_row0, _row1])")

    elif input_count <= 6:
        for index, var in enumerate(input_vars):
            lines.append(f"_c{index} = cv2.resize({var}, ({cell_width}, {cell_height}))")
        for index in range(input_count, 6):
            lines.append(f"_c{index} = np.zeros(({cell_height}, {cell_width}, 3), dtype=np.uint8)")
        lines.append("_row0 = cv2.hconcat([_c0, _c1, _c2])")
        lines.append("_row1 = cv2.hconcat([_c3, _c4, _c5])")
        lines.append(f"{output_var} = cv2.vconcat([_row0, _row1])")

    else:
        for index, var in enumerate(input_vars):
            lines.append(f"_c{index} = cv2.resize({var}, ({cell_width}, {cell_height}))")
        for index in range(input_count, 9):
            lines.append(f"_c{index} = np.zeros(({cell_height}, {cell_width}, 3), dtype=np.uint8)")
        lines.append("_row0 = cv2.hconcat([_c0, _c1, _c2])")
        lines.append("_row1 = cv2.hconcat([_c3, _c4, _c5])")
        lines.append("_row2 = cv2.hconcat([_c6, _c7, _c8])")
        lines.append(f"{output_var} = cv2.vconcat([_row0, _row1, _row2])")

    return "\n".join(lines)


def get_ancestors(node_key, links):
    """
    指定ノードの全祖先ノード（上流ノード）を取得

    Args:
        node_key: ノードキー
        links: リンク情報リスト

    Returns:
        set: 祖先ノードキーの集合
    """
    ancestors = set()
    to_visit = [node_key]
    while to_visit:
        current = to_visit.pop()
        for link in links:
            if link["dst_node"] == current:
                src = link["src_node"]
                if src not in ancestors:
                    ancestors.add(src)
                    to_visit.append(src)
    return ancestors


def replace_template_params(code, node_info, template, setting, node_key, links, nodes,
                           var_mapping, model_file_mapping):
    """
    テンプレートコード内のパラメータを置換

    Args:
        code: テンプレートコード
        node_info: ノード情報
        template: テンプレート定義
        setting: ノード設定
        node_key: ノードキー
        links: リンク情報
        nodes: 全ノード情報
        var_mapping: 変数名マッピング
        model_file_mapping: モデルファイルマッピング

    Returns:
        tuple: (置換後のコード, パラメータ値辞書)
    """
    from json2demo.json_parsing import extract_setting_value, get_linked_param_value

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
            code = code.replace(f"{{{param}}}", value)
            if param == "model":
                model_file = model_file_mapping.get(value, "unknown.onnx")
                code = code.replace("{model_path}", model_file)
        else:
            code = code.replace(f"{{{param}}}", str(value))

    return code, param_values
