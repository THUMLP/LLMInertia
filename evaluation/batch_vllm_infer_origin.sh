datasets=(
    "explicit_test_1000_input"
    "explicit_xugou_test_1000_input"
)

model_name_or_paths=(
    #"/home/youxx/LLaMA-Factory-main/models/Meta-Llama-3.1-8B-Instruct"
    "/home/youxx/LLaMA-Factory-main/models/Qwen2.5-7B-Instruct"
#    "/home/youxx/LLaMA-Factory-main/models/Ministral-8B-Instruct-2410"
#    "/home/youxx/LLaMA-Factory-main/models/DeepSeek-R1-Distill-Qwen-7B"
#    "/home/youxx/LLaMA-Factory-main/models/deepseek-70B"
#    "/home/youxx/LLaMA-Factory-main/models/Meta-Llama-3-70B"
)

templates=(
    #"llama3"
    "qwen"
#    "ministral"
#    "deepseek3"
#    "deepseek3"
#    "llama3"
)

results_base="/home/youxx/LLaMA-Factory-main/results"

for model_idx in "${!model_name_or_paths[@]}"; do
    model_path="${model_name_or_paths[$model_idx]}"
    template="${templates[$model_idx]}"
    model_base=$(basename "$model_path")
    outdir="${results_base}/${model_base}"
    mkdir -p "$outdir"

    for dataset in "${datasets[@]}"; do
        dataset_name=$(basename "$dataset")   # 得到 explicit_nobut_test.jsonl 等

        # 输出文件名示例：/home/youxx/LLaMA-Factory-main/results/Meta-Llama-3.1-8B-Instruct/explicit_nobut_test.jsonl
        save_name="${outdir}/${dataset_name}"
#        if [ -f "$save_name" ]; then
#            echo "$save_name 已存在，跳过。"
#            continue  # 或者 exit 0 或跳到下一轮（如果在循环里）
#        fi
        echo "==== 推理: $model_path | $dataset ===="

        CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/vllm_infer.py \
            --model_name_or_path "$model_path" \
            --dataset "$dataset" \
            --template "$template" \
            --save_name "$save_name"
        infer_code=$?
        if [[ $infer_code -ne 0 ]]; then
            echo "❌ 推理失败: $model_base | $dataset | $model_path" >> "$error_log"
            echo "-----" >> "$error_log"
            continue
        fi
    done
done