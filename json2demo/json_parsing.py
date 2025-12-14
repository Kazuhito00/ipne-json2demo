"""
JSON解析ユーティリティ関数
"""

import json
import re

from json2demo.node_param_mapping import NODE_PARAM_MAPPING


def parse_json(json_path):
    """JSONファイルを解析"""
    with open(json_path, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    return data


def extract_node_info(data):
    """ノード情報を抽出"""
    nodes = {}
    for node_entry in data.get("node_list", []):
        parts = node_entry.split(":")
        node_id = parts[0]
        node_type = parts[1]
        node_key = f"{node_id}:{node_type}"

        node_data = data.get(node_key, {})
        setting = node_data.get("setting", {})

        nodes[node_key] = {
            "id": node_id,
            "type": node_type,
            "setting": setting,
        }
    return nodes


def extract_links(data):
    """リンク情報を抽出"""
    links = []
    for link in data.get("link_list", []):
        src = link[0]
        dst = link[1]

        src_parts = src.split(":")
        dst_parts = dst.split(":")

        links.append({
            "src_node": f"{src_parts[0]}:{src_parts[1]}",
            "src_port": src_parts[3],
            "dst_node": f"{dst_parts[0]}:{dst_parts[1]}",
            "dst_port": dst_parts[3],
        })
    return links


def topological_sort(nodes, links):
    """ノードをトポロジカルソート（処理順序を決定）"""
    in_degree = {node_key: 0 for node_key in nodes}
    adjacency = {node_key: [] for node_key in nodes}

    for link in links:
        src = link["src_node"]
        dst = link["dst_node"]
        if src in nodes and dst in nodes:
            adjacency[src].append(dst)
            in_degree[dst] += 1

    queue = [node for node, degree in in_degree.items() if degree == 0]
    sorted_nodes = []

    while queue:
        node = queue.pop(0)
        sorted_nodes.append(node)
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return sorted_nodes


def get_input_vars(node_key, links, var_mapping):
    """ノードの入力変数名を取得（複数入力対応）"""
    inputs = []
    for link in links:
        if link["dst_node"] == node_key:
            inputs.append({
                "var": var_mapping.get(link["src_node"], "frame"),
                "port": link["dst_port"],
            })

    def port_sort_key(item):
        match = re.search(r"(\d+)$", item["port"])
        return int(match.group(1)) if match else 0

    inputs.sort(key=port_sort_key)
    return inputs


def extract_setting_value(setting, node_id, node_type, param_name, default):
    """設定から値を抽出"""
    node_mapping = NODE_PARAM_MAPPING.get(node_type, {})
    key_suffix = node_mapping.get(param_name)

    if key_suffix:
        full_key = f"{node_id}:{node_type}:{key_suffix}"
        if full_key in setting:
            return setting[full_key]

    for key, value in setting.items():
        if param_name.lower() in key.lower():
            return value

    return default


def get_linked_param_value(node_key, param_name, links, nodes):
    """リンク経由のパラメータ値を取得（IntValue/FloatValueからのリンク）"""
    node_mapping = NODE_PARAM_MAPPING.get(nodes[node_key]["type"], {})
    key_suffix = node_mapping.get(param_name)

    if not key_suffix:
        return None

    match = re.search(r"Input(\d+)", key_suffix)
    if not match:
        return None

    input_port = f"Input{match.group(1)}"

    for link in links:
        if link["dst_node"] == node_key and link["dst_port"] == input_port:
            src_node_key = link["src_node"]
            if src_node_key in nodes:
                src_node = nodes[src_node_key]
                src_type = src_node["type"]
                if src_type in ("IntValue", "FloatValue"):
                    src_setting = src_node["setting"]
                    value_type = "Int" if src_type == "IntValue" else "Float"
                    output_key = f"{src_node['id']}:{src_type}:{value_type}:Output01Value"
                    if output_key in src_setting:
                        return src_setting[output_key]

    return None
