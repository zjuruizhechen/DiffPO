"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""

import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype

def consistency_generate(
    model,
    tokenizer,
    inputs,
    qs_ids,
    num_of_turn,
    max_new_tokens_for_consistency,
    max_new_seq_len,
    max_iter,
    ):
    max_new_tokens = max_new_tokens_for_consistency
    
    itr = 0
    while True:
        if itr == 0:
            input_ids = inputs
            input_masks = torch.ones_like(qs_ids).to(input_ids.device)
            prompt_masks = torch.ones_like(qs_ids).to(input_ids.device)
        else:
            input_masks = torch.ones_like(input_ids).to(input_ids.device)
            prompt_masks = torch.ones_like(qs_ids).to(input_ids.device)
            for j in range(bsz):
                input_masks[j][torch.sum(prompt_masks, dim=-1)[j] + itr * max_new_tokens:] = 0
            
        bsz = input_ids.shape[0]
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        generation = get_jacobian_trajectory(model, tokenizer, input_ids, input_masks, prompt_masks, max_new_tokens, max_iter)
        for j in range(bsz):
            prompt_len = torch.sum(input_masks, dim=-1)
            eos_positions = torch.where(generation[j]==tokenizer.convert_tokens_to_ids("<end_of_turn>"))[0]
            print(eos_positions)
            if len(eos_positions)==num_of_turn*2+1:
                generation[j][prompt_len[j]+ max_new_tokens:] = tokenizer.pad_token_id
                continue
            eos_reached[j] = True
            generation[j, int(eos_positions[num_of_turn*2+1])+1:] = tokenizer.pad_token_id
        itr+=1      
        if all(eos_reached) or itr*max_new_tokens >= max_new_seq_len:
            total_token_len = torch.sum(generation != tokenizer.pad_token_id, dim=-1)
            return generation[:,:total_token_len]

        for j in range(bsz):
            start = torch.sum(prompt_masks, dim=-1)[j] + (itr-1) * max_new_tokens
            end = torch.sum(prompt_masks, dim=-1)[j] + (itr) * max_new_tokens
            input_ids[j][start:end] = generation[j][start:end]

@torch.inference_mode()
def get_jacobian_trajectory(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    prompt_masks,
    max_new_tokens,
    max_iter,
):

    bsz = input_ids.shape[0] 
    prompt_len = [torch.sum(t) for t in attention_mask]
    input_len = [len+max_new_tokens for len in prompt_len]
    max_prompt_len = max(prompt_len)
    total_len = max_prompt_len + max_new_tokens

    # initialize the first point of jacobian trajectory
    if not tokenizer.pad_token_id:
        if "vicuna" in args.model_id or "zephyr" in args.model_id or "mistral" in args.model_id:
            tokenizer.pad_token = '[PAD]'
        else:
            tokenizer.pad_token_id = 128001
    tokens = torch.full((bsz, int(total_len)), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
    for i in range(bsz):
        tokens[i, : input_len[i]] = input_ids[i][: input_len[i]]
    itr = 0
    next_generation = tokens
    generate_attention_mask = torch.full_like(next_generation, 1).to(tokens.device)
    while itr<=max_iter:
        current_generation = next_generation
        with torch.no_grad():
            logits = model(current_generation, generate_attention_mask).logits
        next_generation = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)

        # hold prompt unchanged and update generated tokens
        for i in range(bsz):
            next_generation[i, :] = torch.cat((tokens[i, :prompt_len[i]], next_generation[i, prompt_len[i]-1:total_len-1]), dim=0)
        if torch.all(torch.eq(next_generation, current_generation)).item():
            print(f"Iteration steps: {itr}")
            return next_generation # right generation is saved twice so we delete the last element of trajectory list
        itr+=1
    print(f"Iteration steps: {itr}")
    return next_generation

def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    use_consistency_decoding,
    max_new_tokens_for_consistency,
    revision,
    max_iter,
):

    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                use_consistency_decoding=use_consistency_decoding,
                max_new_tokens_for_consistency=max_new_tokens_for_consistency,
                revision=revision,
                max_iter=args.max_iter,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    use_consistency_decoding,
    max_new_tokens_for_consistency,
    max_iter,
):
    model, tokenizer = load_model(
        model_path,
        revision=revision,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            # print(model_id)
            turns = []
            answers = []

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<end_of_turn>")
            ]

            for j in range(len(question["turns"])):
                qs = question["turns"][j]

                print(qs)
                print('-'*50)

                turns.append({"role": "user", "content": qs})

                qs_idsx = tokenizer.apply_chat_template(
                    turns,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)

                ans = question["choices"][0]["turns"][j]


                turns.append({"role": "assistant", "content": ans})

                input_txt = tokenizer.apply_chat_template(
                    turns, tokenize=False, add_generation_prompt=False)

                import random
                reserve = random.randint(0, args.reserve_tokens)

                input_ids = tokenizer(
                    input_txt,
                    return_tensors="pt",
                    padding="max_length",  # 填充到固定长度
                    truncation=True,  # 截断超过固定长度的部分
                    max_length=qs_idsx.size(1)+max_new_token+reserve  # 设置固定长度
                )['input_ids'].to(dtype=torch.int64)

                qs_ids = input_ids[:, :qs_idsx.size(1)+reserve]



                if temperature < 1e-4:
                    do_sample = False
                else:
                    do_sample = True

                # some models may error out when generating long outputs
                # try:
                if use_consistency_decoding:
                    output_ids = consistency_generate(
                        model,
                        tokenizer,
                        input_ids,
                        qs_ids,
                        num_of_turn=j,
                        max_new_tokens_for_consistency=max_new_tokens_for_consistency,
                        max_new_seq_len=max_new_token,
                        max_iter=args.max_iter,
                    )
                else:
                    output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        do_sample=do_sample,
                        temperature=temperature,
                        eos_token_id=terminators,
                        max_new_tokens=max_new_token,
                    )
                if model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(qs_idsx[0]) :]


                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                    skip_special_tokens=True,
                )


                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()


                print('--------------- output ----------------')
                print(output)
                print('--------------- output ends ----------------')

                answers.append(output)

            choices.append({"index": i, "turns": answers})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
                "category": question["category"],
                "turns": question["turns"],
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID. Default: cllm/consistency-llm-7b-sharegpt48k",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--save-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--original-answer-id", type=str, default=None, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=256,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--use_consistency_decoding",
        action='store_false',
        help="Whether to use consistency decoding",
    )
    parser.add_argument(
        "--max_new_tokens_for_consistency",
        type=int,
        default=32,
        help="The n-gram for consistency decoding.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="The n-gram for consistency decoding.",
    )
    parser.add_argument(
        "--reserve_tokens",
        type=int,
        default=12,
        help="The n-gram for consistency decoding.",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"./question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"./DiffuPO_answer/{args.save_id}.jsonl"


    original_answer_file = f"./origin_answer/{args.original_answer_id}.jsonl"

    print(f"Output to {answer_file}")
    print(args.use_consistency_decoding)
    print(args.model_path)

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=original_answer_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        use_consistency_decoding=args.use_consistency_decoding,
        max_new_tokens_for_consistency = args.max_new_tokens_for_consistency,
        max_iter=args.max_iter,
    )

    reorg_answer_file(answer_file)

