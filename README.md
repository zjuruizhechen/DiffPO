
# DIFFPO: Diffusion-styled Preference Optimization for Efficient Inference-Time Alignment of Large Language Models


This repository contains the official implementation of the paper DIFFPO: Diffusion-styled Preference Optimization for Efficient
Inference-Time Alignment of Large Language Models.

Acknowledgement: This repository is built based on https://github.com/hao-ai-lab/Consistency_LLM.

## Installation
1. Environment setup:
```
conda create -n DiffuPO python=3.10
conda activate DiffuPO

pip install -r requirements.txt
pip install flash-attn==2.4.1
```

# Usage

We have released DiffuPO models:

| Model      | Path                                        |
|------------|---------------------------------------------|
| DiffuPO-9B | https://huggingface.co/RuizheChen/DiffPO-9B |
| DiffuPO-8B | https://huggingface.co/RuizheChen/DiffPO-8B |
| DiffuPO-2B | https://huggingface.co/RuizheChen/DiffPO-2B |

## Inference 
```
cd DiffuPO/eval/mt-bench

CUDA_VISIBLE_DEVICES=0 python gen_DiffuPO_answer.py --model-path RuizheChen/DiffPO-9B --model-id gemma --original-answer-id llama-3-it-vanilla --save-id llama-3-it-DiffPO-9B-256-256 --max_new_tokens_for_consistency 256 --max-new-token 256
```

## Training

The training module will be released soon.

## Evaluation

### MT-bench

To evaluate the models on MT-Bench, please use the FastChat LLM Judge package. (https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge#mt-bench)

To generate GPT-4 judgments for base models, run:
```
python gen_judgment.py --model-list llama-3-it-vanilla llama-3-SFT-vanilla mistral-it-vanilla zephyr-SFT-vanilla --parallel 1
```
To generate GPT-4 judgments for DiffuPO, run:
```
python gen_judgment.py --model-list llama-3-it-DiffPO-9B-256-256 --parallel 1
```
Show results
```
python show_result.py --model-list llama-3-it-DiffPO-9B-256-256
```

### AlpacaEval 2

To evaluate the models on AlpacaEval 2, please use the alpaca-eval package. (https://github.com/tatsu-lab/alpaca_eval)


