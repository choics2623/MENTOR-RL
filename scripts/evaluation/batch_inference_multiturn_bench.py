#!/usr/bin/env python3
"""
Multi-turn batch inference script using VLLM for HuggingFace datasets
Copied multi-turn logic from src/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
"""

import os
import sys

# CRITICAL: Force online mode for HuggingFace Hub to download datasets
# Remove any offline mode environment variables
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)
os.environ.pop('HF_DATASETS_OFFLINE', None)

# Set online mode explicitly
os.environ['HF_HUB_OFFLINE'] = '0'

import json
import argparse
import requests
import re
from typing import List, Dict, Any
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Import functions from reward score file
from src.verl.utils.reward_score.mentor_sparse import (
    last_boxed_only_string,
    remove_boxed,
    unified_judge,
    compute_score_with_format
)
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-turn batch inference with VLLM using qwen3_native template")
    parser.add_argument("--dataset_name", type=str, default="YOUR_HF_DATASET",
                       help="HuggingFace dataset name or benchmark shortname (e.g., 'gsm8k', 'math', 'olympiad')")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-1.7B",
                       help="Path to the model")
    parser.add_argument("--hf_token", type=str, default="YOUR_HF_TOKEN_HERE",
                       help="HuggingFace token")
    parser.add_argument("--benchmark_category", type=str, default=None,
                       help="Override category name for benchmark (default: extracted from dataset_name)")
    parser.add_argument("--output_file", type=str, default="qwen3_multiturn_results.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for inference")
    parser.add_argument("--max_tokens", type=int, default=12000,
                       help="Maximum response length")
    parser.add_argument("--max_model_len", type=int, default=32768,
                       help="Maximum model context length")
    # parser.add_argument("--max_model_len", type=int, default=40960,
    #                     help="Maximum model context length")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                       help="GPU memory utilization")
    parser.add_argument("--max_num_batched_tokens", type=int, default=131072,
                       help="Maximum number of batched tokens per iteration")
    parser.add_argument("--enable_chunked_prefill", action="store_true",
                       help="Enable chunked prefill")
    parser.add_argument("--enable_prefix_caching", action="store_true", default=True,
                       help="Enable prefix caching")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process")
    # Multi-turn settings from train script
    parser.add_argument("--max_turns", type=int, default=5,
                       help="Maximum number of turns for multi-turn conversation")
    parser.add_argument("--sandbox_url", type=str, default="http://0.0.0.0:2623",
                       help="Sandbox URL for tool execution")
    parser.add_argument("--categories", type=str, nargs='+', default=None,
                       help="Specific categories to evaluate (e.g., aime24 aime25 nvidia-AceReason-Math)")

    return parser.parse_args()

# === Copied from vllm_rollout_spmd.py ===
def format_tool_call(tool_call_str: str):
    """Convert JSON function call description to Python executable code string."""
    try:
        call_json = json.loads(tool_call_str)
        func_name = call_json['name']
        arguments = call_json.get('arguments', {})

        args_str = ', '.join(f"{k}={repr(v)}" for k, v in arguments.items())
        return f"{func_name}({args_str})"
    except Exception as e:
        return f"Parse tool call failed: {e}"

def validate_tool_calls(output_str):
    start_tags = re.findall(r'<tool_call>', output_str)
    end_tags = re.findall(r'</tool_call>', output_str)

    if len(start_tags) != len(end_tags):
        return False

    start_positions = [m.start() for m in re.finditer(r'<tool_call>', output_str)]
    end_positions = [m.start() for m in re.finditer(r'</tool_call>', output_str)]

    for start, end in zip(start_positions, end_positions):
        if start >= end:
            return False

    return True

def extract_tool_calls(output_str):
    if not validate_tool_calls(output_str):
        return []

    try:
        pattern = r'<tool_call>((?:(?!</tool_call>).)*)</tool_call>'
        matches = re.finditer(pattern, output_str, re.DOTALL)

        return [match.group(1).strip() for match in matches]
    except Exception as e:
        return []

def execute_tool_call(env, call, sandbox_url):
    """Execute single tool call via sandbox - copied from spmd"""
    url = f'{sandbox_url}/execute'

    call_str = format_tool_call(call)
    if call_str.startswith("Parse tool call failed"):
        return call_str

    try:
        # Use same payload format as spmd
        data = {
            'env': env,
            'call': call_str
        }
        response = requests.post(url, json=data, timeout=10)
        if response.status_code != 200:
            return f"error: {response.status_code}"
        response = response.json()
        ret_str = ''
        if response.get('result'):
            ret_str += f'result: \n{response["result"]}\n'
        if response.get('stdout'):
            ret_str += f'stdout: \n{response["stdout"]}\n'
        if response.get('stderr'):
            ret_str += f'stderr: \n{response["stderr"]}\n'
        return ret_str.strip() if ret_str else "No output"
    except Exception as e:
        return f"Execution error: {str(e)}"

def execute_tool_call_with_session(env, call, sandbox_url, session):
    """Execute single tool call via sandbox with session reuse (faster)"""
    url = f'{sandbox_url}/execute'

    call_str = format_tool_call(call)
    if call_str.startswith("Parse tool call failed"):
        return call_str

    try:
        data = {
            'env': env,
            'call': call_str
        }
        response = session.post(url, json=data, timeout=3)  # Reduced timeout
        if response.status_code != 200:
            return f"error: {response.status_code}"
        response = response.json()
        ret_str = ''
        if response.get('result'):
            ret_str += f'result: \n{response["result"]}\n'
        if response.get('stdout'):
            ret_str += f'stdout: \n{response["stdout"]}\n'
        if response.get('stderr'):
            ret_str += f'stderr: \n{response["stderr"]}\n'
        return ret_str.strip() if ret_str else "No output"
    except Exception as e:
        return f"Execution error: {str(e)}"

def batch_execute(env_list: List[str], tool_calls_list: List[List[str]], sandbox_url: str):
    """Batch execute tool calls using thread pool with session reuse"""
    # Create session for connection pooling with larger pool size
    from requests.adapters import HTTPAdapter
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    def exe_tool_call(env, call):
        return execute_tool_call_with_session(env, call, sandbox_url, session)

    all_tasks = []
    for env_idx, (env, tool_calls) in enumerate(zip(env_list, tool_calls_list)):
        for call_idx, tool_call in enumerate(tool_calls):
            all_tasks.append((env, tool_call))

    all_task_indices = []
    for env_idx, (env, tool_calls) in enumerate(zip(env_list, tool_calls_list)):
        for call_idx, tool_call in enumerate(tool_calls):
            all_task_indices.append((env_idx, call_idx))

    with ThreadPoolExecutor(max_workers=64) as executor:
        future_to_index = {executor.submit(exe_tool_call, env, call): i
                          for i, (env, call) in enumerate(all_tasks)}

        results = [None] * len(all_tasks)
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    session.close()

    results_list = [[None for _ in range(len(tool_calls_list[i]))] for i, _ in enumerate(env_list)]
    for (env_idx, call_idx), result in zip(all_task_indices, results):
        results_list[env_idx][call_idx] = result

    return results_list
# === End copy from vllm_rollout_spmd.py ===

def get_question_field(sample: Dict) -> str:
    """Get question field from sample (handle different dataset formats)"""
    if 'question' in sample:
        return sample['question']
    elif 'problem' in sample:
        return sample['problem']
    else:
        raise ValueError(f"No question/problem field found in sample: {sample.keys()}")

def get_answer_field(sample: Dict) -> str:
    """Get answer field from sample (handle different dataset formats)"""
    if 'answer' in sample:
        return sample['answer']
    elif 'gold_label' in sample:
        return sample['gold_label']
    else:
        return 'unknown'

def extract_model_answer(conversation: str) -> str:
    """Extract the final boxed answer from conversation"""
    try:
        boxed_string = last_boxed_only_string(conversation)
        if boxed_string:
            return remove_boxed(boxed_string).strip()
    except Exception:
        pass
    return ""

def check_answer_correctness(conversation: str, gold_label: str, tokenizer) -> tuple[bool, str]:
    """Check if answer is correct using unified_judge only (no EOS token check)"""
    try:
        # Extract model answer
        model_answer = extract_model_answer(conversation)
        if not model_answer:
            return False, "No boxed answer found"

        # Use unified judge only (no EOS token check)
        is_correct = unified_judge(model_answer, gold_label)
        return is_correct, f"Unified judge result: {model_answer} vs {gold_label}"
    except Exception as e:
        return False, f"Error in answer checking: {e}"

def convert_to_new_format(question: str, conversation: str, gold_label: str,
                         model_name: str, dataset_name: str, tokenizer) -> dict:
    """Convert conversation to new format with messages array"""

    # Extract model answer and check correctness
    model_answer = extract_model_answer(conversation)

    # Format gold_label for reward_model field
    if isinstance(gold_label, str):
        ground_truth = [gold_label]
    elif isinstance(gold_label, list):
        ground_truth = gold_label
    else:
        ground_truth = [str(gold_label)]

    # Check correctness using the first ground truth value
    is_correct, check_reason = check_answer_correctness(conversation, ground_truth[0], tokenizer)

    # Parse conversation into messages
    messages = []

    # Split conversation by role markers including system
    parts = re.split(r'<\|im_start\|>(user|assistant|system)', conversation)

    current_role = None
    for i, part in enumerate(parts):
        if part in ['user', 'assistant', 'system']:
            current_role = part
        elif current_role and part.strip():
            # Clean up the content
            content = part.replace('<|im_end|>', '').strip()
            if content:
                messages.append({
                    "role": current_role,
                    "content": content
                })

    return {
        "question": question,
        "messages": messages,
        "reward_model": {
            "ground_truth": ground_truth,
            "style": "rule"
        },
        "model_answer": model_answer,
        "is_correct": is_correct,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "ability": "kd"
    }

def format_prompt_qwen3_native(question: str, func_schemas: list, tokenizer) -> str:
    """Format prompt using qwen3_native style (Qwen3 native tool calling)"""
    # Add instruction about boxed answer format to user input
    formatted_user_input = f"Question: {question}\n\nIf you have got the answer, enclose it within \\boxed{{}} with latex format."

    # Use native tool calling with enable_thinking
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': formatted_user_input}],
        tools=func_schemas,
        add_generation_prompt=True,
        tokenize=False
    )

def process_batch_multiturn(samples: List[Dict], llm: LLM, tokenizer, sampling_params: SamplingParams,
                           max_turns: int, sandbox_url: str, model_name: str, dataset_name: str,
                           default_extra_info: Dict = None) -> List[Dict]:
    """Process batch of samples with multi-turn conversation (like spmd)"""

    batch_size = len(samples)
    print(f"Starting batch processing for {batch_size} samples")

    # Prepare initial data for each sample
    conversations = []
    envs = []
    sample_metadata = []

    for sample in samples:
        question = get_question_field(sample)

        # Parse func_schemas and env from extra_info
        try:
            if 'extra_info' in sample and sample['extra_info']:
                if isinstance(sample['extra_info'], str):
                    extra_info = json.loads(sample['extra_info'])
                else:
                    extra_info = sample['extra_info']
                func_schemas = extra_info.get('func_schemas', [])
                env = extra_info.get('env', '')
            else:
                # Use default extra_info from YOUR_HF_DATASET
                if default_extra_info:
                    func_schemas = default_extra_info.get('func_schemas', [])
                    env = default_extra_info.get('env', '')
                else:
                    func_schemas = []
                    env = ''
        except:
            if default_extra_info:
                func_schemas = default_extra_info.get('func_schemas', [])
                env = default_extra_info.get('env', '')
            else:
                func_schemas = []
                env = ''

        # Initial prompt
        conversation = format_prompt_qwen3_native(question, func_schemas, tokenizer)
        conversations.append(conversation)
        envs.append(env)
        sample_metadata.append({
            'original_sample': sample,
            'turns_used': 0,
            'final_response': '',
            'tools_used': [],  # Track valid tools used during conversation
            'invalid_tools_used': []  # Track invalid tool calls
        })

    # Track active samples (samples that still need processing)
    active_indices = list(range(batch_size))
    # Track remaining tokens for each sample (like spmd)
    curr_max_tokens = [sampling_params.max_tokens] * batch_size
    # Track samples to skip due to length issues
    skipped_samples = set()

    # Multi-turn loop (like spmd)
    for turn in range(max_turns):
        if not active_indices:
            break

        print(f"Turn {turn + 1}: Processing {len(active_indices)} active samples")

        # Prepare batch for active samples only
        active_conversations = []
        active_max_tokens = []
        valid_active_indices = []

        # Check conversation length and skip if too long
        for idx in active_indices:
            conv = conversations[idx]
            conv_tokens = tokenizer.encode(conv)

            # If conversation is too long, skip this sample entirely
            max_conv_len = 38000  # Leave some room for generation
            if len(conv_tokens) > max_conv_len:
                print(f"Skipping sample {idx} due to length ({len(conv_tokens)} tokens > {max_conv_len})")
                skipped_samples.add(idx)
                continue

            active_conversations.append(conv)
            active_max_tokens.append(curr_max_tokens[idx])
            valid_active_indices.append(idx)

        # Update active_indices to only include valid ones
        active_indices = valid_active_indices

        # If no valid samples left, break the turn loop
        if not active_indices:
            print(f"No valid samples left to process in turn {turn + 1}")
            break

        # Generate responses with updated sampling params
        spmd_sampling_params = SamplingParams(
            n=1,
            max_tokens=min(8192, max(active_max_tokens)) if active_max_tokens else 8192,
            stop_token_ids=[151644],  # <|im_start|> stop token
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
        )

        # Handle prompt length errors - try batch generation with error handling
        try:
            outputs = llm.generate(active_conversations, spmd_sampling_params)
        except ValueError as e:
            if "longer than the maximum model length" in str(e):
                print(f"Warning: Some prompts exceeded max length, processing individually...")
                # Process each conversation individually to identify problematic ones
                outputs = []
                skipped_indices = []

                for idx, conv in enumerate(active_conversations):
                    try:
                        output = llm.generate([conv], spmd_sampling_params)
                        outputs.extend(output)
                    except ValueError as length_error:
                        if "longer than the maximum model length" in str(length_error):
                            sample_idx = active_indices[idx]
                            print(f"Skipping sample {sample_idx} due to length ({len(tokenizer.encode(conv))} tokens)")
                            skipped_samples.add(sample_idx)
                            # Create dummy output for skipped sample
                            from vllm.outputs import RequestOutput, CompletionOutput
                            dummy_output = RequestOutput(
                                request_id=f"dummy_{idx}",
                                prompt="",
                                prompt_token_ids=[],
                                prompt_logprobs=None,
                                finished=True,
                                outputs=[CompletionOutput(
                                    index=0,
                                    text="[Skipped due to length]",
                                    token_ids=[],
                                    cumulative_logprob=0.0,
                                    logprobs=None,
                                    finish_reason="length"
                                )]
                            )
                            outputs.append(dummy_output)
                            skipped_indices.append(idx)
                        else:
                            raise length_error
            else:
                raise e

        # Process outputs and determine next active samples
        new_active_indices = []
        tool_calls_batch = []
        call_indices = []
        env_batch = []

        for i, output in enumerate(outputs):
            # Safely get sample index with bounds checking
            if i >= len(active_indices):
                print(f"Warning: Output index {i} exceeds active_indices length {len(active_indices)}, skipping")
                continue

            sample_idx = active_indices[i]

            # Skip if already marked as skipped
            if sample_idx in skipped_samples:
                continue

            try:
                generated_text = output.outputs[0].text
            except (IndexError, AttributeError) as e:
                print(f"Error accessing output for sample {sample_idx}: {e}, marking as skipped")
                skipped_samples.add(sample_idx)
                continue

            # Update conversation
            conversations[sample_idx] += generated_text
            sample_metadata[sample_idx]['final_response'] = generated_text
            sample_metadata[sample_idx]['turns_used'] = turn + 1

            # Extract tool calls
            tool_calls = extract_tool_calls(generated_text)

            if tool_calls:
                # This sample needs another turn
                tool_calls_batch.append(tool_calls)
                call_indices.append(sample_idx)
                env_batch.append(envs[sample_idx])
                new_active_indices.append(sample_idx)

                # Record tools used for this sample
                for call in tool_calls:
                    try:
                        call_data = json.loads(call)
                        tool_name = call_data.get('name')
                        if tool_name:
                            sample_metadata[sample_idx]['tools_used'].append(tool_name)
                        else:
                            sample_metadata[sample_idx]['invalid_tools_used'].append(call)
                    except:
                        sample_metadata[sample_idx]['invalid_tools_used'].append(call)

        # Execute tool calls if any
        if tool_calls_batch:
            print(f"Executing {len(tool_calls_batch)} tool call batches")
            tool_responses_list = batch_execute(env_batch, tool_calls_batch, sandbox_url)

            # Add tool responses to conversations
            for idx, (sample_idx, tool_calls, tool_responses) in enumerate(zip(call_indices, tool_calls_batch, tool_responses_list)):
                tool_response_str = ""
                for call, response in zip(tool_calls, tool_responses):
                    tool_response_str += f"<tool_response>{call}\n{response}\n</tool_response>\n"

                # Format as user message (like spmd)
                tool_response_str = "\n<|im_start|>user\n" + tool_response_str + "<|im_end|>"

                # Add assistant start for next turn
                tool_response_str += "\n<|im_start|>assistant\n<think>"

                conversations[sample_idx] += tool_response_str

        # Update curr_max_tokens for used tokens (like spmd)
        for i, output in enumerate(outputs):
            sample_idx = active_indices[i]
            used_tokens = len(output.outputs[0].token_ids)
            curr_max_tokens[sample_idx] = max(0, curr_max_tokens[sample_idx] - used_tokens)

        # Update active indices for next turn
        active_indices = new_active_indices

    # Create results in new format (exclude skipped samples)
    results = []
    for i, metadata in enumerate(sample_metadata):
        # Skip samples that were too long
        if i in skipped_samples:
            print(f"Excluding sample {i} from results (skipped due to length)")
            continue

        sample = metadata['original_sample']
        # Get gold label - could be a single value or list
        gold_label = get_answer_field(sample)

        result = convert_to_new_format(
            question=get_question_field(sample),
            conversation=conversations[i],
            gold_label=gold_label,
            model_name=model_name,
            dataset_name=dataset_name,
            tokenizer=tokenizer
        )
        # Add additional metadata
        result["turns_used"] = metadata['turns_used']
        result["category"] = sample.get("category", "unknown")
        result["tools_used"] = metadata['tools_used']
        result["invalid_tools_used"] = metadata['invalid_tools_used']
        result["extra_info"] = sample.get("extra_info", {})
        results.append(result)

    print(f"Processed {len(results)} samples successfully, skipped {len(skipped_samples)} due to length")
    return results

def process_single_sample_multiturn(sample: Dict, llm: LLM, tokenizer, sampling_params: SamplingParams,
                                   max_turns: int, sandbox_url: str, default_extra_info: Dict = None) -> Dict:
    """Process single sample with multi-turn conversation"""

    question = get_question_field(sample)

    # Parse func_schemas and env from extra_info
    try:
        if 'extra_info' in sample and sample['extra_info']:
            if isinstance(sample['extra_info'], str):
                extra_info = json.loads(sample['extra_info'])
            else:
                extra_info = sample['extra_info']
            func_schemas = extra_info.get('func_schemas', [])
            env = extra_info.get('env', '')
        else:
            # Use default extra_info from YOUR_HF_DATASET
            if default_extra_info:
                func_schemas = default_extra_info.get('func_schemas', [])
                env = default_extra_info.get('env', '')
            else:
                func_schemas = []
                env = ''
    except:
        if default_extra_info:
            func_schemas = default_extra_info.get('func_schemas', [])
            env = default_extra_info.get('env', '')
        else:
            func_schemas = []
            env = ''

    # Initial prompt
    conversation = format_prompt_qwen3_native(question, func_schemas, tokenizer)
    full_conversation = conversation

    for turn in range(max_turns):
        # Generate response
        outputs = llm.generate([conversation], sampling_params)
        generated_text = outputs[0].outputs[0].text

        full_conversation += generated_text

        # Extract tool calls
        tool_calls = extract_tool_calls(generated_text)

        if not tool_calls:
            # No more tool calls, conversation ends
            break

        # Execute tool calls
        tool_responses_list = batch_execute([env], [tool_calls], sandbox_url)
        tool_responses = tool_responses_list[0]

        # Add tool responses as user message (copied from spmd format)
        tool_response_str = ""
        for call, response in zip(tool_calls, tool_responses):
            tool_response_str += f"<tool_response>{call}\n{response}\n</tool_response>\n"

        # Format as user message (like spmd line 559)
        tool_response_str = "\n<|im_start|>user\n" + tool_response_str + "<|im_end|>"

        # Add assistant start for next turn (like spmd line 564)
        tool_response_str += "\n<|im_start|>assistant\n<think>"

        full_conversation += tool_response_str
        conversation = full_conversation  # Continue from the full conversation

    # Create result entry
    result = {
        "question": get_question_field(sample),
        "model_response": generated_text,  # Last generated text
        "conversation": full_conversation,  # Full conversation including all turns
        "gold_label": get_answer_field(sample),
        "category": sample.get("category", "unknown"),
        "extra_info": sample.get("extra_info", {}),
        "turns_used": turn + 1
    }

    return result

def main():
    args = parse_args()

    # Set HuggingFace token
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    print(f"Loading dataset: {args.dataset_name}")

    # Load dataset
    dataset_split = "train"
    if args.max_samples:
        dataset_split = f"train[:{args.max_samples}]"

    # Check if dataset_name is a local path
    if os.path.exists(args.dataset_name):
        print(f"Loading from local path: {args.dataset_name}")
        full_dataset = load_from_disk(args.dataset_name)
        dataset = full_dataset["train"]
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    # Special handling for nvidia/AceReason-Math
    elif args.dataset_name == "nvidia/AceReason-Math":
        dataset = load_dataset('json',
                              data_files='https://huggingface.co/datasets/nvidia/AceReason-Math/resolve/main/math.jsonl',
                              split=dataset_split)
    else:
        dataset = load_dataset(args.dataset_name, split=dataset_split)
    print(f"Dataset loaded with {len(dataset)} samples")

    # Filter by categories if specified
    if args.categories:
        print(f"Filtering for categories: {args.categories}")
        filtered_samples = []
        for sample in dataset:
            if 'category' in sample and sample['category'] in args.categories:
                filtered_samples.append(sample)

        if filtered_samples:
            from datasets import Dataset
            dataset = Dataset.from_list(filtered_samples)
            print(f"Filtered dataset: {len(dataset)} samples matching categories: {args.categories}")
        else:
            print(f"Warning: No samples found for categories: {args.categories}")
            return

    # Load default extra_info from YOUR_HF_DATASET if current dataset doesn't have it
    default_extra_info = None
    if args.dataset_name != "YOUR_HF_DATASET":
        try:
            print("Loading default extra_info from YOUR_HF_DATASET...")
            changsu_dataset = load_dataset("YOUR_HF_DATASET", split="train[:1]")
            changsu_sample = changsu_dataset[0]
            if 'extra_info' in changsu_sample:
                if isinstance(changsu_sample['extra_info'], str):
                    default_extra_info = json.loads(changsu_sample['extra_info'])
                else:
                    default_extra_info = changsu_sample['extra_info']
                print(f"Loaded default extra_info with {len(default_extra_info.get('func_schemas', []))} functions")
        except Exception as e:
            print(f"Could not load default extra_info: {e}")
            default_extra_info = None

    # Initialize tokenizer
    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    # Validate max_num_batched_tokens vs max_model_len (like spmd)
    if args.max_num_batched_tokens < args.max_model_len and args.enable_chunked_prefill:
        raise ValueError(f'Enable chunked prefill, max_num_batched_tokens ({args.max_num_batched_tokens}) is smaller than max_model_len ({args.max_model_len}), '
                        'please increase max_num_batched_tokens or disable chunked prefill')

    # Initialize VLLM
    print(f"Initializing VLLM with model: {args.model_path}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Max num batched tokens: {args.max_num_batched_tokens}")
    print(f"Max model len: {args.max_model_len}")
    print("This may take several minutes for model loading and GPU initialization...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        # max_num_batched_tokens=args.max_num_batched_tokens,
        # enable_chunked_prefill=args.enable_chunked_prefill,
        # enable_prefix_caching=args.enable_prefix_caching,
        trust_remote_code=True,
        seed=args.seed,
        # enforce_eager=True,  # Use eager mode for simplicity
        # disable_custom_all_reduce=True,  # Like spmd
        # enable_expert_parallel=True
    )
    print("VLLM model loaded successfully!")

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    print(f"Starting multi-turn batch inference...")
    print(f"Max turns: {args.max_turns}")
    print(f"Sandbox URL: {args.sandbox_url}")
    print(f"Output file: {args.output_file}")

    # Process samples in batches (like spmd)
    total_samples = len(dataset)
    processed = 0

    # Check for existing output file to resume from
    start_batch = 0
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                existing_lines = sum(1 for _ in f)
            start_batch = existing_lines // args.batch_size
            processed = existing_lines
            print(f"Found existing output file with {existing_lines} samples, resuming from batch {start_batch + 1}")
        except:
            print(f"Could not read existing file, starting from beginning")
            start_batch = 0
            processed = 0

    # Open file in append mode if resuming, write mode if starting fresh
    file_mode = 'a' if start_batch > 0 else 'w'

    with open(args.output_file, file_mode, encoding='utf-8') as f:
        for batch_idx in range(start_batch, (total_samples + args.batch_size - 1) // args.batch_size):
            i = batch_idx * args.batch_size
            batch_end = min(i + args.batch_size, total_samples)
            batch_samples = [dataset[j] for j in range(i, batch_end)]

            print(f"Processing batch {batch_idx + 1}/{(total_samples + args.batch_size - 1)//args.batch_size} "
                  f"(samples {i+1}-{batch_end}/{total_samples})")

            # Process batch with multi-turn
            results = process_batch_multiturn(
                batch_samples, llm, tokenizer, sampling_params,
                args.max_turns, args.sandbox_url, args.model_path, args.dataset_name, default_extra_info
            )

            # Write results to JSONL file
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()  # Ensure data is written immediately

            processed += len(results)
            avg_turns = sum(r['turns_used'] for r in results) / len(results)
            print(f"Processed {processed}/{total_samples} samples (avg turns: {avg_turns:.1f})")

    print(f"Multi-turn batch inference completed! Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()