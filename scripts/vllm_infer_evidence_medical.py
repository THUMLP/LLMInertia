import json
from typing import Optional
import fire
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer

if is_vllm_available():
    from vllm import LLM, SamplingParams

# ------- 步骤1 -------
def build_step1_prompt(input_text):
    f_extract_prompt = """
Instruction: Carefully read the following medical input and list all important entities (such as diseases, symptoms, drugs, examinations, tests, etc.) mentioned, in the format below.
- Output entities as a comma-separated English list. No extra words.

Example:
Input: The patient reports experiencing a persistent cough and fever for three days. Chest X-ray was performed with no abnormalities detected.
Output: Cough, Fever, Chest X-ray

Now, process:
Input: {input}
Output:
"""
    return f_extract_prompt.format(input=input_text)

# ------- 步骤2 -------
def build_step2_prompt(entities_text):
    f_probe_prompt = """
Instruction: For each given entity, output three entities that are most closely related to it. List all results in the form of tuple. See the format and example below.

Format:
Entities: Entity1, Entity2
Output:
(Entity1,  Co1), (Entity1,  Co2), (Entity1,  Co3)
(Entity2,  Co1), (Entity2,  Co2), (Entity2,  Co3)

Example:
Entities: Headache, Cough
Output: 
(Headache, Dizziness), (Headache,  Nausea), (Headache,  Chest pain)
(Cough,  Shortness of breath), (Cough,  Sore throat), (Cough,  Fatigue)

Entities: {entities}
Output:
"""
    return f_probe_prompt.format(entities=entities_text)


def build_step3_prompt(question, note_text):
    f_step3_prompt = """
Instruction: Please answer the following question and strictly respond in the format: 【Conclusion】: Yes/No.
Here are two examples:

Example 1:
Question: The patient's main complaint during this visit is Headache, which can sometimes indicate Hypertension, although no clinical or laboratory abnormalities related to Hypertension have been found. Does Hypertension exist or hold?
Note: Headache and Hypertension may not necessarily be related. Please carefully examine the original text and make a thorough judgment.
Answer: 【Conclusion】No.

Example 2:
Question: The patient has been experiencing persistent cough for several days, which can sometimes suggest Pneumonia; imaging results showed clear lungs with signs of infection Pneumonia. Does Pneumonia exist or hold?
Note: Cough and Pneumonia may not necessarily be related. Please carefully examine the original text and make a thorough judgment.
Answer: 【Conclusion】Yes.


Question: {question}
Note: {note_text}
Answer: 
"""
    return f_step3_prompt.format(question=question, note_text=note_text)

# ========= 通用 batch 推理函数 =========
def batch_generate(llm, tokenizer, sampling_params, prompts):
    results = []
    outputs = llm.generate(prompts, sampling_params)
    for out in outputs:
        if out.outputs and len(out.outputs) > 0:
            results.append(out.outputs[0].text.strip())
        else:
            results.append("")
    return results

def find_entity_pairs_in_input(input_text, cooccur_triples):
    """
    cooccur_triples: 形如 '(Headache, Dizziness), (Headache, Nausea)' 的字符串
    返回一个列表，每个元素是实体对 (A, B)，要求A和B都在原文中
    """
    import re
    # 解析字符串为实体对列表
    pairs = re.findall(r"\((.*?),\s*(.*?)\)", cooccur_triples)
    # 标准化文本匹配
    input_lower = input_text.lower()
    pairs_in_input = []
    for a, b in pairs:
        a = a.strip()
        b = b.strip()
        # 只简单lower()比较（如需更强可用fuzzy match或正则）
        if a.lower() in input_lower and b.lower() in input_lower:
            pairs_in_input.append((a, b))
    return pairs_in_input

def build_entity_pair_prompt(pairs_in_input):
    if pairs_in_input:
        advice = []
        for a, b in pairs_in_input:
            advice.append(f"{a} and {b} may not necessarily be related. Please examine the original text carefully and make a thorough judgment.")
        return " ".join(advice)
    else:
        return (
            "Please read the original text carefully, especially paying attention to information that is often assumed by default but not actually given in the question. "
            "Watch out for typical 'traps' or details that are easily overlooked or misjudged."
        )

# ========= 多步链式自由文本推理 =========
def vllm_chain_multi_steps(llm, tokenizer, sampling_params, input_texts):
    # Step 1
    step1_prompts = [build_step1_prompt(txt) for txt in input_texts]
    step1_outputs = batch_generate(llm, tokenizer, sampling_params, step1_prompts)
    #print(step1_outputs)

    # Step 2
    step2_prompts = [build_step2_prompt(txt) for txt in step1_outputs]
    step2_outputs = batch_generate(llm, tokenizer, sampling_params, step2_prompts)
    #print(step2_outputs)

    # ------- 新增逻辑从这里开始 -------
    step3_prompts = []
    for txt, co in zip(input_texts, step2_outputs):
        pairs_in_input = find_entity_pairs_in_input(txt, co)
        advice_prompt = build_entity_pair_prompt(pairs_in_input)
        prompt = build_step3_prompt(txt,advice_prompt)
        step3_prompts.append(prompt)
    # ------- 新增逻辑到此结束 -------

    # Step 3
    print(step3_prompts)
    step3_outputs = batch_generate(llm, tokenizer, sampling_params, step3_prompts)
    print(step3_outputs)

    return step1_outputs, step2_outputs, step3_prompts, step3_outputs

# ========= 推理主入口 =========
def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "base_test",
    dataset_dir: str = "data",
    template: str = "llama3",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "xxx.jsonl",
    temperature: float = 0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    batch_size: int = 8,
):
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )
    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
        "gpu_memory_utilization": 0.8,
        "max_num_seqs": 4,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}
    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)
    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,
        top_k=generating_args.top_k or -1,
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )

    samples = list(dataset_module["train_dataset"])
    total = len(samples)
    with open(save_name, "w", encoding="utf-8") as f:
        for start in range(0, total, batch_size):
            this_batch = samples[start : start + batch_size]
            question_texts = [
                tokenizer.decode(s["input_ids"], skip_special_tokens=skip_special_tokens).strip()
                for s in this_batch
            ]
            ref_labels = [
                tokenizer.decode(
                    list(filter(lambda x: x != IGNORE_INDEX, s["labels"])),
                    skip_special_tokens=skip_special_tokens
                ).strip()
                for s in this_batch
            ]
            step1_batch, step2_batch, step3_input, step3_batch = vllm_chain_multi_steps(
                llm, tokenizer, sampling_params, question_texts
            )
            for i in range(len(this_batch)):
                output = {
                    "prompt": question_texts[i],
                    "step1_entities": step1_batch[i],
                    "step2_cooccur": step2_batch[i],
                    "argumented_input": step3_input[i],
                    "predict": step3_batch[i],
                    "label": ref_labels[i],
                }
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
                f.flush()
                print(f"完成第{start + i + 1}题")
    print(f"全部生成已保存在: {save_name}")

if __name__ == "__main__":
    fire.Fire(vllm_infer)