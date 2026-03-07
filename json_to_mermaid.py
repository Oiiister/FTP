import json


def json_to_mermaid(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    triplets = data['triplets']

    mermaid_code = ["graph TD"]  # TD 表示从上到下的结构

    # 样式定义：区分不同类型的事件
    mermaid_code.append("    classDef basic fill:#e1f5fe,stroke:#01579b,stroke-width:2px;")
    mermaid_code.append("    classDef inter fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;")
    mermaid_code.append("    classDef top fill:#ffcdd2,stroke:#b71c1c,stroke-width:2px;")

    for t in triplets:
        # 定义关系逻辑（OR/AND）
        relation_symbol = "---"
        if t['relation'] == 'jointly_resultsIn':
            rel_label = "|AND|"
        else:
            rel_label = "|OR|"

        # 生成节点连接代码
        mermaid_code.append(f"    {t['subject_name']} --{rel_label}--> {t['object_name']}")

        # 为节点赋予样式
        style = "basic" if t['subject_type'] == "BasicEvent" else "inter"
        mermaid_code.append(f"    class {t['subject_name']} {style}")

    # 为 TopEvent 赋予特殊样式
    mermaid_code.append("    class 控制单元（CU）故障,驱动系统停机 top")

    return "\n".join(mermaid_code)


# 执行并输出
print(json_to_mermaid(r'C:\Users\oyste\Desktop\fault_tree_extractor\data\output\test_triplets.json'))