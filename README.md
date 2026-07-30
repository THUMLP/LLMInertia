# LLMInertia: Adaptive Counter-Inertial Reasoning to Improve Evidence Faithfulness in Large Language Models

This repository contains the implementation for the ICML 2026 paper:  
**"LLMInertia: Adaptive Counter-Inertial Reasoning to Improve Evidence Faithfulness in Large Language Models".**

---

##  Requirements


The code is based on the [LlamaFactory framework.](https://github.com/hiyouga/LlamaFactory)



## Induction Experiments

### Train the Model with Inductive Data

```bash
bash examples/train_full/batch_train_multi_model_full.sh
```

- You can set the base model, base model template, and dataset to batch save model results.
- Remember to configure the yaml and save the model name.
- Add the dataset information in data/dataset_info.json.

---

### Inference and Test Results

- Inference with the original (non-induced) model:

  ```bash
  bash evaluation/batch_vllm_infer_origin.sh
  ```

- Inference with the induced models:

  ```bash
  bash evaluation/batch_infer_eval_savesteps_full.sh
  ```

## Hallucination Mitigation

```bash
python scripts/vllm_infer_evidence_medical.py
```
