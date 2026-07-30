#!/bin/bash


# 模型（实验大目录）
model_name_or_paths=(
    "/home/youxx/LLaMA-Factory-main/savesnew/faithful/Meta-Llama-3.1-8B-Instruct"
    "/home/youxx/LLaMA-Factory-main/savesnew/faithful/Qwen2.5-7B-Instruct"
#    "/home/youxx/LLaMA-Factory-main/savesnew/faithful/Ministral-8B-Instruct-2410"
#    "/home/youxx/LLaMA-Factory-main/savesnew/faithful/DeepSeek-R1-Distill-Qwen-7B"
)


templates=(
    "llama3"
    "qwen"
#    "ministral"
#    "deepseek3"
)


results_base="/home/youxx/LLaMA-Factory-main/results"
summary_csv="$results_base/llama-evidence.csv"
error_log="$results_base/error_eval.log"
> "$summary_csv"
> "$error_log"

# 【1】表头添加‘函数名’一列 ★★★ （已修改）
echo "模型名称,数据集,执行脚本,审题能力相对下降,诱导率(正确->错误),相对错误增幅,诱导前准确率,诱导后准确率,诱导前类型分布,诱导后类型分布" > "$summary_csv"

shopt -s nullglob   # 不存在也不报错for就不会进

for model_idx in "${!model_name_or_paths[@]}"; do
    model_dir="${model_name_or_paths[$model_idx]}"
    template="${templates[$model_idx]}"
    model_base=$(basename "$model_dir")
    outdir="$results_base/$model_base"
    mkdir -p "$outdir"


    # 遍历每个任务子目录
    for subdir in "$model_dir"/*/; do
        subdir_name=$(basename "$subdir")

        # 判断xugou相关
        if [[ "$subdir_name" == *xugou* ]]; then
            pre_file="explicit_xugou_test_1000_input"
        else
            pre_file="explicit_test_1000_input"
        fi

        infer_file="$outdir/$pre_file"

        # 遍历子目录内的每个checkpoint
        for ckpt in "$subdir"*/; do
            ckpt_name=$(basename "$ckpt")

            # == SPECIAL MARK 【2】分别定义要跑的脚本及它们的可区分名
            declare -A infer_scripts
            infer_scripts["vllm_infer_evidence"]="scripts/vllm_infer_evidence.py"
            #infer_scripts["vllm_infer"]="scripts/vllm_infer.py"
            #infer_scripts["vllm_infer_prompt_baseline"]="scripts/vllm_infer_prompt_baseline_faithful.py"

            # == SPECIAL MARK 【3】遍历每个脚本分别跑
            for script_name in "${!infer_scripts[@]}"; do
                script_file="${infer_scripts[$script_name]}"

                # 生成保存文件名，包含脚本名防止覆盖
                save_name="$outdir/${subdir_name}_${ckpt_name}_${script_name}.jsonl"
#                if [ -f "$save_name" ]; then
#                    echo "$save_name 已存在，跳过。"
#                    continue
#                fi

                echo "==== 推理[$script_file]: $ckpt | $pre_file ===="
                CUDA_VISIBLE_DEVICES=4,5,6,7 python "$script_file" \
                    --model_name_or_path "$ckpt" \
                    --dataset "$pre_file" \
                    --template "$template" \
                    --save_name "$save_name"
                infer_code=$?

                if [[ $infer_code -ne 0 ]]; then
                    echo "❌ 推理失败: ${model_base}_${subdir_name}_${ckpt_name} | $pre_file | $ckpt | $script_file" >> "$error_log"
                    echo "-----" >> "$error_log"
                    continue
                fi

                # 【4】评估时带入 script_name
                # evaluation/eval_overthink_ch.py 需要适配接收这个 script_name 作为参数，并写入 csv
                python evaluation/eval_overthink.py "$infer_file" "$save_name" "$script_name" "${model_base}_${subdir_name}_${ckpt_name}" "$pre_file"  "$summary_csv"
                eval_code=$?
                if [[ $eval_code -ne 0 ]]; then
                    echo "❌ 评估失败: ${model_base}_${subdir_name}_${ckpt_name} | $pre_file | $save_name | $script_name" >> "$error_log"
                    echo "-----" >> "$error_log"
                    continue
                fi

                echo "✅ Done: ${model_base}_${subdir_name}_${ckpt_name} | $pre_file | $script_name"
            done
            # == END 回到checkpoint循环
        done
    done
done