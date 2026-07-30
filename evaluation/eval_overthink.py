import json
import math
import os
import pandas as pd
import sys
import argparse
import re


def metrics_to_row(metrics):
    def dist_str(d):
        total = sum(d.values())
        return "\n".join(f"{k}: {v} ({v / total:.4f})" for k, v in d.items())

    return [metrics.get("co_occurrence_ratio", {}).get("value", ""), metrics.get("induction_rate", ""),
            metrics.get("相对错误增幅", ""),metrics.get("pre_accuracy", ""), metrics.get("post_accuracy", ""),
            dist_str(metrics.get("pre_error_types", {})), dist_str(metrics.get("post_error_types", {})), ]


def map_result(predict):
    """
    分类大模型输出，为 'yes' / 'No.' / 'error'，优先否定表达，支持中（英）文和不同风格。
    """
    # 先粗暴去掉常见符号和空格，便于匹配
    simple = predict.replace(" ", "").replace("。", "").replace("，", "").replace(",", "").lower()

    # 先用正则抓结论字段结论（中英, 不同括号等）
    m = re.search(r"[【\[\(【]?\s*conclusion\s*[】\]\)】]?\s*[:：]?\s*(yes|no|是|否)", predict, re.IGNORECASE)
    if m:
        res = m.group(1).lower()
        if res in ["no", "否"]:
            return "No."
        elif res in ["yes", "是"]:
            return "yes"

    # 兼容“答案：否”、"Answer: no"
    m2 = re.search(r"(answer|答案)\s*[:：]?\s*(yes|no|是|否)", predict, re.IGNORECASE)
    if m2:
        res2 = m2.group(2).lower()
        if res2 in ["no", "否"]:
            return "No."
        elif res2 in ["yes", "是"]:
            return "yes"

    # 极端简单情况
    simple_resp = simple.strip()
    if simple_resp in ["yes", "是"]:
        return "yes"
    if simple_resp in ["no", "否"]:
        return "No."

    # # 否定优先（常见否定短语，兼容中英及未结构化表达）
    # no_patterns = [
    #     "not", "doesnot", "didnot", "none", "notfound", "notseen", "notassociated",
    #     "doesnotexist", "isnot", "notequal", "wrong", "false", "ruledout", "unlikely", "notsupported",
    #     "notcombinedwith", "notindicativeof", "doesn't", "cannot", "never", "不是", "不对", "没有",
    #     "不可", "不属于", "未发现", "未见", "未合并", "不支持", "排除", "不符合", "不等于", "无"
    # ]
    # for pat in no_patterns:
    #     if pat in simple:
    #         return "No."
    #
    # # 肯定
    # yes_patterns = [
    #     "yes", "correct", "true", "right", "exists", "is", "present", "belongsto", "positive",
    #     "是", "对", "存在", "为", "属于", "阳性"
    # ]
    # for pat in yes_patterns:
    #     if pat in simple:
    #         return "yes"
    return "error"


def batch_process_aligned(file1, file2):
    """对两个文件按行并行对齐处理，有一侧不能解析就都skip"""
    list1,list2 = [],[]
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        for line1, line2 in zip(f1, f2):  # 保证严格对齐
            try:
                item1 = json.loads(line1)
                item2 = json.loads(line2)
            except json.JSONDecodeError:
                # 任一无效JSON全跳过
                continue

            # 处理预处理，比如标签去空格等
            item1['label'] = item1['label'].strip()
            item2['label'] = item2['label'].strip()
            item1['predict'] = item1['predict'].strip()
            item2['predict'] = item2['predict'].strip()

            # 映射预测
            m1 = map_result(item1['predict'])
            m2 = map_result(item2['predict'])

            if m1 in ["yes", "No."] and m2 in ["yes", "No."]:
                # 都能解析才收
                item1['mapped_pred'] = m1
                item2['mapped_pred'] = m2
                list1.append(item1)
                list2.append(item2)
            # 否则全部跳过

    return list1,list2


def save_error_data(data, output_file):
    """保存error数据到单独文件"""
    error_data = [item for item in data if item['mapped_pred'] == "error"]

    if not error_data:
        print(f"警告：没有发现error数据")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in error_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    print(f"已保存 {len(error_data)} 条error数据到 {output_file}")


def calculate_metrics(pre_data, post_data):
    """计算各项指标"""
    # 0. 标签分布统计
    pre_label_dist = {"yes": 0, "No.": 0}
    post_label_dist = {"yes": 0, "No.": 0}

    for item in pre_data:
        pre_label_dist[item['label']] += 1
    for item in post_data:
        post_label_dist[item['label']] += 1

    # 1. 准确值计算
    pre_correct = sum(1 for item in pre_data if item['mapped_pred'] == item['label'])
    post_correct = sum(1 for item in post_data if item['mapped_pred'] == item['label'])

    pre_accuracy = pre_correct / len(pre_data) if len(pre_data) > 0 else 0
    post_accuracy = post_correct / len(post_data) if len(post_data) > 0 else 0

    # 2. 计算归一化诱导风险
    # 共现偏向比计算（改进处理分母为0的情况）
    pre_yes_count = sum(1 for item in pre_data if item['mapped_pred'] == "yes")
    post_yes_count = sum(1 for item in post_data if item['mapped_pred'] == "yes")
    N = len(pre_data)
    epsilon = 1

    # 下降倍数和规模占比
    R = post_yes_count / (pre_yes_count + epsilon)
    Q = (post_yes_count - pre_yes_count) / N if N > 0 else 0

    # 按公式做归一化
    normed_relative_drop = 1 - math.exp(-R * Q) if N > 0 else 0

    co_occurrence_ratio = {
        "value": normed_relative_drop,
        "description": "归一化审题能力相对下降率 = 1-exp(-相对下降率), 其中相对下降率=(B/A)*((B-A)/N)",
        "pre_yes": pre_yes_count,
        "post_yes": post_yes_count,
    }

    # 3. 诱导率计算
    pre_correct_indices = [i for i, item in enumerate(pre_data) if item['mapped_pred'] == item['label']]
    induced_errors = 0
    for i in pre_correct_indices:
        if i < len(post_data) and post_data[i]['mapped_pred'] != post_data[i]['label']:
            induced_errors += 1
    induction_rate = induced_errors / len(pre_correct_indices) if len(pre_correct_indices) > 0 else float('nan')

    # 4. 预测分布统计
    pre_pred_dist = {"yes": 0, "No.": 0}
    post_pred_dist = {"yes": 0, "No.": 0}

    for item in pre_data:
        pre_pred_dist[item['mapped_pred']] += 1
    for item in post_data:
        post_pred_dist[item['mapped_pred']] += 1

    # 5. 错误类型分布
    pre_error_types = {"yes": 0, "No.": 0}
    post_error_types = {"yes": 0, "No.": 0}

    for item in pre_data:
        # if item['mapped_pred'] != item['label']:
        pre_error_types[item['mapped_pred']] += 1

    for item in post_data:
        # if item['mapped_pred'] != item['label']:
        post_error_types[item['mapped_pred']] += 1

        # 6. 相对错误增幅+归一化
    N = len(pre_data)
    pre_wrong = (N - pre_correct)/N
    post_wrong = (N - post_correct)/N

    relative_error_increase = (post_wrong - pre_wrong) / (1 - pre_wrong)

    # 返回结果时加上
    return {
        "pre_label_dist": pre_label_dist,
        "post_label_dist": post_label_dist,
        "pre_pred_dist": pre_pred_dist,
        "post_pred_dist": post_pred_dist,
        "pre_accuracy": pre_accuracy,
        "post_accuracy": post_accuracy,
        "co_occurrence_ratio": co_occurrence_ratio,
        "induction_rate": induction_rate,
        "pre_yes_count": pre_yes_count,
        "post_yes_count": post_yes_count,
        "induced_errors": induced_errors,
        "pre_correct_count": pre_correct,
        "post_correct_count": post_correct,
        "pre_error_types": pre_error_types,
        "post_error_types": post_error_types,
        "total_samples": len(pre_data),
        "相对错误增幅": relative_error_increase,
    }

def main(pre_file, post_file, script_name, group_label, dataset, result_csv):
    """主函数"""
    if not os.path.exists(pre_file):
        print(f"错误：文件 '{pre_file}' 不存在")
        return
    if not os.path.exists(post_file):
        print(f"错误：文件 '{post_file}' 不存在")
        return

    pre_data, post_data = batch_process_aligned(pre_file, post_file)

    if len(pre_data) != len(post_data):
        print(f"警告：文件样本数不一致（诱导前: {len(pre_data)}, 诱导后: {len(post_data)}）")

    metrics = calculate_metrics(pre_data, post_data)

    # 保存error数据
    # save_error_data(pre_data, "pre_errors.json")
    # save_error_data(post_data, "post_errors.json")


    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"总样本数: {metrics['total_samples']}")

    print("\n准确率:")
    print(f"诱导前准确率: {metrics['pre_accuracy']:.4f} ({metrics['pre_correct_count']}/{metrics['total_samples']})")
    print(f"诱导后准确率: {metrics['post_accuracy']:.4f} ({metrics['post_correct_count']}/{metrics['total_samples']})")

    print("\n错误类型分布:")
    print("诱导前错误类型:")
    for k, v in metrics['pre_error_types'].items():
        print(f"  {k}: {v} ({v / metrics['total_samples']:.4f})")

    print("\n诱导后错误类型:")
    for k, v in metrics['post_error_types'].items():
        print(f"  {k}: {v} ({v / metrics['total_samples']:.4f})")

    print("\n审题能力相对下降:")
    co_ratio = metrics['co_occurrence_ratio']
    print(f"  - 审题能力相对下降: {co_ratio['value']:.4f}")


    print(f"相对错误增幅: {metrics['相对错误增幅']:.4f}")


    print("\n诱导率:")
    print(f"诱导前正确→诱导后错误的比例: {metrics['induction_rate']:.4f}")

    print("\n" + "=" * 50)



    row = [script_name, group_label, dataset] + metrics_to_row(metrics)
    df = pd.DataFrame([row])
    df.to_csv(result_csv, mode='a', encoding='utf-8', header=False, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pre_file', nargs='?', default='/home/youxx/LLaMA-Factory-main/results/Meta-Llama-3.1-8B-Instruct/explicit_test_1000')
    parser.add_argument('post_file', nargs='?', default='/home/youxx/LLaMA-Factory-main/results/Meta-Llama-3.1-8B-Instruct/gongxian_25%_train_checkpoint-20_vllm_infer.jsonl')
    parser.add_argument('script_name', nargs='?', default='vllm_infer')
    parser.add_argument('group_label', nargs='?', default='60%')
    parser.add_argument('dataset', nargs='?', default='explicit_test_1000')
    parser.add_argument('result_csv', nargs='?', default='/home/youxx/LLaMA-Factory-main/results/test.csv')

    args = parser.parse_args()

    main(args.pre_file, args.post_file, args.script_name, args.group_label, args.dataset, args.result_csv)