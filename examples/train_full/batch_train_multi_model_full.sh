#!/bin/bash

datasets=(
    "gongxian_25%_train"
    "gongxian_50%_train"
    "gongxian_75%_train"
    "gongxian_100%_train"
    "gongxian_xugou25%_train"
    "gongxian_xugou50%_train"
    "gongxian_xugou75%_train"
    "gongxian_xugou100%_train"
)

model_name_or_paths=(
    "/home/youxx/LLaMA-Factory-main/models/Meta-Llama-3.1-8B-Instruct"
    "/home/youxx/LLaMA-Factory-main/models/Qwen2.5-7B-Instruct"
    #"/home/youxx/LLaMA-Factory-main/models/Ministral-8B-Instruct-2410"
    #"/home/youxx/LLaMA-Factory-main/models/DeepSeek-R1-Distill-Qwen-7B"
)


templates=(
    #"llama3"
    #"qwen"
#    "ministral"
#    "deepseek3"
)

output_base="/home/youxx/LLaMA-Factory-main/savesnew/faithful"

for model_idx in "${!model_name_or_paths[@]}"; do
    model_path="${model_name_or_paths[$model_idx]}"
    template="${templates[$model_idx]}"
    model_base=$(basename "$model_path")

    for dataset in "${datasets[@]}"; do
        outdir="$output_base/${model_base}/${dataset}"
        yaml_file="tmp_${model_base}_${dataset}.yaml"
        echo "======================================================================"
        echo "Training with model: $model_path, template: $template, dataset: $dataset, output_dir: $outdir"

        # 复制模板yaml文件
        #cp /home/youxx/LLaMA-Factory-main/examples/train_full/deepseek_lora_pt.yaml "$yaml_file"
        cp /home/youxx/LLaMA-Factory-main/examples/train_full/llama3_full_pt.yaml "$yaml_file"
        # 替换dataset
        sed -i "s/^dataset: .*/dataset: ${dataset}/g" "$yaml_file"
        # 替换output_dir
        sed -i "s|^output_dir: .*|output_dir: ${outdir}|g" "$yaml_file"
        # 替换model_name_or_path
        sed -i "s|^model_name_or_path: .*|model_name_or_path: ${model_path}|g" "$yaml_file"
        # 替换template
        sed -i "s/^template: .*/template: ${template}/g" "$yaml_file"

        set +e
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train "$yaml_file"
        if [ $? -ne 0 ]; then
            echo "❌ Training failed for model $model_base on dataset $dataset, skipping..."
        else
            echo "✅ Training finished for model $model_base on dataset $dataset"
        fi
        set -e

        rm -f "$yaml_file"

    done
done