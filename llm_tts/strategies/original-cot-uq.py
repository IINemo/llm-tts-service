import json
import logging
import re
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F

from config import args
from src.model.llama2_predict import predict, model_init
from src.format.get_cot_prompt import get_cot_prompt
from src.format.get_step_exact_tokens import get_step_exact_tokens

from utils import load_data, print_exp, setup_log, is_effectively_empty, find_subsequence_position, step_exacts_2_list, \
                    parse_response_to_dict, find_token_indices, is_word_in_sentence, match_final_answer_token_ids 

def llama_inference_refining():
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    log = setup_log(args)

    if args.dataset in ["hotpotQA", "2WikimhQA"]:
        question, answer, ids, types = load_data(args)
    else:
        question, answer, ids = load_data(args)
    model, tokenizer, device = model_init(args)
    model.eval()
    
    for idx, q in enumerate(tqdm(question[], total=len(question[]))):
        with open(f"{args.output_path}/output_v1.json", "a", encoding="utf-8") as f:

            if args.dataset in ["gsm8k", "svamp", "ASDiv"]:
                a = answer[idx]
                id = ids[idx]
            else:
                a = answer[idx]       
                id = ids[idx]
                t = types[idx]

            log.debug(f"##### This is the --{idx + 1}th-- Question #####")

            cot_prompt = get_cot_prompt(args, q)

            inputs = tokenizer(cot_prompt, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            try_time = 0
            while try_time < args.try_times:
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens = args.max_length_cot, 
                    temperature=args.temperature, 
                    pad_token_id=tokenizer.eos_token_id, 
                    return_dict_in_generate=True, 
                    output_scores=True,
                )

                generated_ids = outputs.sequences[0][len(inputs["input_ids"][0]):-1]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                llm_answer, steps_dict, response = parse_response_to_dict(response)
                if generated_ids.size(0) >= args.max_length_cot:
                    log.debug(f'New Reasoning Tokens Are Too Much, Current try is {try_time + 1}')
                    try_time += 1
                elif generated_ids.size(0) == 0:
                    log.debug(f'New Reasoning Tokens Are Null, Current try is {try_time + 1}')
                    try_time += 1
                elif llm_answer is None or llm_answer in ['', ' ']:
                    log.debug(f'New Reasoning Tokens Are None, Current try is {try_time + 1}')
                    try_time += 1
                else:
                    response_tokens = tokenizer.tokenize(response)
                    response_token_ids = tokenizer.convert_tokens_to_ids(response_tokens)
                    original_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
                    probabilities = [{i: p for i, p in enumerate(prob[0]) if p > 0} for prob in [torch.softmax(score, dim=1).tolist() for score in outputs.scores]]

                    final_answer_probabilities = {}
                    final_answer_token_ids = {}
                    answer_start_indice, answer_token_ids = match_final_answer_token_ids(args, original_tokens, response_tokens, generated_ids)
                    if answer_start_indice == None:
                        log.debug(f'Cannot locate the Final Answer, Current try is {try_time + 1}')
                        try_time += 1
                        continue
                    answer_probs = []
                    flag = False
                    for j, token_id in enumerate(answer_token_ids):
                        idxx = j + answer_start_indice
                        if token_id not in probabilities[idxx].keys():                       
                            flag = True
                            break
                        answer_probs.append(probabilities[idxx][token_id])
                    if flag:
                        log.debug(f'Cannot locate the Final Answer Token Probability, Current try is {try_time + 1}')
                        try_time += 1
                        continue
                    final_answer_probabilities[llm_answer] = answer_probs
                    final_answer_token_ids[llm_answer] = answer_token_ids.tolist()

                    exacts_prompt = get_step_exact_tokens(args, q, response)
                    exact_response = predict(args, exacts_prompt, model, tokenizer)

                    if "NO ANSWER" in exact_response:
                        log.debug(f'Exact Tokens Have NO ANSWER, Current try is {try_time + 1}')
                        try_time += 1
                        continue
                    if not step_exacts_2_list(exact_response):
                        log.debug(f'Exact Tokens Have no contribution scores, Current try is {try_time + 1}')
                        try_time += 1
                        continue
                    exact_response, keywords_list, contributions_list = step_exacts_2_list(exact_response)
                    if len(keywords_list) == 0:
                        log.debug(f'Cannot Exract Effective Keywords, Current try is {try_time + 1}')
                        try_time += 1
                        continue
                    
                    if len(steps_dict) > len(keywords_list):
                        log.debug(f'Len of keywords list doesn\'t match the len of step dict, Current try is {try_time + 1}')
                        try_time += 1
                        continue

                    keywords_probabilities = {}
                    keywords_contributions = {}
                    keywords_token_ids = {}
                    for step_idx, (step_name, step_text) in enumerate(steps_dict.items()):
                        # # Skip the Final Answer
                        keywords = keywords_list[step_idx]
                        contributions = contributions_list[step_idx]
                        if len(keywords) == 1 and keywords[0] == 'NO ANSWER':
                            continue
                        step_tokens = tokenizer.tokenize(step_text)
                        processed_step_tokens = [token[1:] if token.startswith('Ġ') or token.startswith('▁')  else token for token in step_tokens]
                        step_token_ids = tokenizer.convert_tokens_to_ids(step_tokens)
                        start_position = find_subsequence_position(step_token_ids[1:-2], generated_ids) - 1
                        step_token_ids = generated_ids[start_position: start_position + len(step_tokens)]
                        keywords_probabilities_dict = {}
                        keywords_contributions_dict = {}
                        keywords_token_ids_dict = {}
                        for keyword_idx, keyword in enumerate(keywords):

                            keyword_probs = []
                            keyword_token_ids = []
                            if is_word_in_sentence(step_text, keyword) is not True:
                                log.debug(f"\n{step_name}-Keyword-{keyword_idx} Does not appear in the Step Text")
                                continue
                            keyword_token_start_idx, keyword_token_end_idx = find_token_indices(processed_step_tokens, keyword)
                            keyword_token_ids = generated_ids[start_position + keyword_token_start_idx: start_position + keyword_token_end_idx + 1]
                            keyword_token_ids = keyword_token_ids.data.cpu().numpy()

                            for j, token_id in enumerate(keyword_token_ids):
                                idxx = start_position + keyword_token_start_idx + j
                                keyword_probs.append(probabilities[idxx][token_id])
                            keywords_probabilities_dict[keyword] = keyword_probs
                            keywords_contributions_dict[keyword] = int(contributions[keyword_idx])
                            keywords_token_ids_dict[keyword] = keyword_token_ids.tolist()
                            
                        keywords_probabilities[step_name] = keywords_probabilities_dict
                        keywords_contributions[step_name] = keywords_contributions_dict
                        keywords_token_ids[step_name] = keywords_token_ids_dict

                    if is_effectively_empty(keywords_probabilities):
                        log.debug(f'Token Probability from All Steps are All None, Current try is {try_time + 1}')
                        try_time += 1
                        continue

                    if args.dataset in ["gsm8k", "svamp", "ASDiv"]:
                        formatted_data = {
                            "id": id,
                            "question": q,
                            "correct answer": a,
                            "llm response": response,
                            "llm answer": llm_answer,
                            "llm answer token probability": final_answer_probabilities, 
                            "step-wise keywords": exact_response,
                            "keyword token probability": keywords_probabilities,
                            "keyword contribution": keywords_contributions,
                            "keyword token ids": keywords_token_ids,
                            "final answer token ids": final_answer_token_ids,
                        }
                    else:
                        formatted_data = {
                            "id": id,
                            "question": q,
                            "correct answer": a,
                            "type": t,
                            "llm response": response,
                            "llm answer": llm_answer,
                            "llm answer token probability": final_answer_probabilities, 
                            "step-wise keywords": exact_response,
                            "keyword token probability": keywords_probabilities,
                            "keyword contribution": keywords_contributions,
                            "keyword token ids": keywords_token_ids,
                            "final answer token ids": final_answer_token_ids,
                        }
                    f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")
                    break
        if try_time >= args.try_times:
            log.debug(f'#####The Following Question:#####\n{q}\nHas no Meaningful Answer & Explanations, Record and Skip')
            error_dir = f"{args.output_path}/error_questions"
            if not os.path.exists(error_dir):
                os.makedirs(error_dir) 
            with open(f"{args.output_path}/error_questions/output_v1.json", "a", encoding="utf-8") as f:
                if args.dataset in ["gsm8k", "svamp", "ASDiv"]:
                    formatted_data = {
                        "id": id,
                        "question": q,
                        "correct answer": a,
                        "llm response": response,
                        "llm answer": llm_answer
                    }
                # hotpotQA 2WikiMHQA
                else:
                    formatted_data = {
                        "id": id,
                        "question": q,
                        "correct answer": a,
                        "type": t,
                        "llm response": response,
                        "llm answer": llm_answer
                    }
                f.write(json.dumps(formatted_data, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    print_exp(args) 

    if args.model_engine in ["llama3-1_8B", "llama2-13b"]:
        llama_inference_refining()
    else:
        raise ValueError(f"Invalid model engine: {args.model_engine}")
        