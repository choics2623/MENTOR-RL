# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

import torch
from typing import Any, Dict, List, Callable
import numpy as np
from verl import DataProto
from collections import Counter, defaultdict
from functools import partial
import re
import json

# Import unified judge for consistent accuracy calculation
try:
    from verl.utils.reward_score.mentor_sparse import unified_judge, remove_boxed, last_boxed_only_string
    UNIFIED_JUDGE_AVAILABLE = True
except ImportError:
    UNIFIED_JUDGE_AVAILABLE = False

# Global cumulative counters for invalid tool tracking
_CUMULATIVE_INVALID_TOOLS = 0
_CUMULATIVE_TOTAL_RESPONSES = 0


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _analyze_tool_usage(response_text: str, valid_tool_names: set = None) -> Dict[str, Any]:
    """
    Analyze tool usage patterns in a response text.
    Returns metrics about tool calling behavior.
    
    Args:
        response_text: The response text to analyze
        valid_tool_names: Set of valid tool names from schema (optional)
    """
    # Count <tool_call> tags
    tool_call_count = response_text.count('<tool_call>')
    
    # Check if <think> tag is used
    has_think_tag = '<think>' in response_text
    
    # Extract tool names from tool calls
    tool_pattern = r'<tool_call>(.*?)</tool_call>'
    tool_calls = re.findall(tool_pattern, response_text, re.DOTALL)
    
    used_tools = []
    invalid_tools = []
    
    for tool_call in tool_calls:
        tool_name = None
        try:
            # Parse JSON to extract tool name
            tool_json = json.loads(tool_call.strip())
            if 'name' in tool_json:
                tool_name = tool_json['name']
        except:
            # If JSON parsing fails, try regex to find name
            name_match = re.search(r'"name":\s*"([^"]+)"', tool_call)
            if name_match:
                tool_name = name_match.group(1)
        
        if tool_name:
            if valid_tool_names and tool_name not in valid_tool_names:
                invalid_tools.append(tool_name)
            else:
                used_tools.append(tool_name)
    
    # Calculate diversity (entropy of tool usage)
    tool_diversity = 0.0
    if used_tools:
        tool_counts = Counter(used_tools)
        total_calls = sum(tool_counts.values())
        
        for count in tool_counts.values():
            p = count / total_calls
            if p > 0:
                tool_diversity -= p * np.log2(p)
    
    # Count each tool usage
    tool_usage_counts = Counter(used_tools)
    invalid_tool_counts = Counter(invalid_tools)
    
    return {
        'tool_call_count': tool_call_count,
        'unique_tools_count': len(set(used_tools)),
        'used_tools': used_tools,
        'invalid_tools': invalid_tools,
        'tool_usage_counts': tool_usage_counts,
        'invalid_tool_counts': invalid_tool_counts,
        'has_think_tag': has_think_tag,
        'tool_diversity': tool_diversity,
        'most_used_tool': Counter(used_tools).most_common(1)[0][0] if used_tools else None,
        'invalid_tool_call_count': len(invalid_tools)
    }


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def _extract_model_answer(response_text: str) -> str:
    """Extract the final answer from model response using same method as reward function"""
    # Use same extraction method as reward function
    if UNIFIED_JUDGE_AVAILABLE:
        try:
            boxed_string = last_boxed_only_string(response_text)
            if boxed_string:
                return remove_boxed(boxed_string)
        except Exception:
            pass
    
    # Fallback to simple regex (may not handle nested braces correctly)
    import re
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, response_text)
    
    if matches:
        return matches[-1].strip()  # Return the last boxed answer
    
    return ""

def _normalize_answer(s):
    """Normalize answer same as reward function for consistency"""
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _get_f1_score_for_accuracy(prediction: str, ground_truths):
    """Calculate F1 score same as reward function for consistency"""
    from collections import Counter
    
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif not isinstance(ground_truths, list):
        ground_truths = [str(ground_truths)]
    else:
        ground_truths = [str(gt) for gt in ground_truths]
    
    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = _normalize_answer(prediction)
        normalized_ground_truth = _normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])
    
    return final_metric['f1']

def _check_answer_correctness(model_answer: str, ground_truth) -> bool:
    """Check if model answer matches ground truth using same method as reward function"""
    if not model_answer:
        return False
    
    # Use unified judge if available (same as reward function)
    if UNIFIED_JUDGE_AVAILABLE:
        try:
            return unified_judge(model_answer, ground_truth)
        except Exception:
            pass
    
    # Fallback to F1 scoring method
    f1_score = _get_f1_score_for_accuracy(model_answer, ground_truth)
    return f1_score > 0

def _compute_tool_metrics(batch: DataProto, tokenizer=None) -> Dict[str, Any]:
    """
    Compute tool calling related metrics from the batch.
    """
    if tokenizer is None:
        # Return empty metrics if no tokenizer available
        return {}
    
    # Define valid tool names (from Mentor-KD math tools schema)
    valid_tool_names = {
        'add', 'subtract', 'multiply', 'divide', 'sum_numbers',
        'floor', 'ceil', 'round_number', 'power', 'sqrt', 
        'abs_value', 'modulo'
    }
    
    tool_call_counts = []
    unique_tools_counts = []
    think_tag_usage = []
    tool_diversities = []
    all_used_tools = []
    all_invalid_tools = []
    tool_usage_rates = []
    invalid_tool_call_counts = []
    correctness_indicators = []
    
    # Aggregate counters for individual tools
    batch_tool_counter = Counter()
    batch_invalid_counter = Counter()
    
    responses = batch.batch['responses']  # (batch_size, seq_len)
    
    for i in range(len(responses)):
        try:
            # Decode response to text
            response_ids = responses[i]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
            
            # Analyze tool usage with validation
            tool_analysis = _analyze_tool_usage(response_text, valid_tool_names)
            
            tool_call_counts.append(tool_analysis['tool_call_count'])
            unique_tools_counts.append(tool_analysis['unique_tools_count'])
            think_tag_usage.append(1.0 if tool_analysis['has_think_tag'] else 0.0)
            tool_diversities.append(tool_analysis['tool_diversity'])
            invalid_tool_call_counts.append(tool_analysis['invalid_tool_call_count'])
            
            all_used_tools.extend(tool_analysis['used_tools'])
            all_invalid_tools.extend(tool_analysis['invalid_tools'])
            tool_usage_rates.append(1.0 if tool_analysis['tool_call_count'] > 0 else 0.0)
            
            # Check answer correctness
            data_item = batch[i]
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            model_answer = _extract_model_answer(response_text)
            is_correct = _check_answer_correctness(model_answer, ground_truth)
            correctness_indicators.append(1 if is_correct else 0)
            
            # Update batch counters
            batch_tool_counter.update(tool_analysis['tool_usage_counts'])
            batch_invalid_counter.update(tool_analysis['invalid_tool_counts'])
            
        except Exception as e:
            # If decoding fails, append zeros
            tool_call_counts.append(0)
            unique_tools_counts.append(0)
            think_tag_usage.append(0.0)
            tool_diversities.append(0.0)
            tool_usage_rates.append(0.0)
            invalid_tool_call_counts.append(0)
            correctness_indicators.append(0)
    
    # Calculate batch-level metrics
    most_used_tool = batch_tool_counter.most_common(1)[0][0] if batch_tool_counter else "none"
    most_invalid_tool = batch_invalid_counter.most_common(1)[0][0] if batch_invalid_counter else "none"
    
    # Base metrics
    metrics = {
        'tool_metrics/call_count/mean': np.mean(tool_call_counts),
        'tool_metrics/call_count/max': np.max(tool_call_counts),
        'tool_metrics/call_count/min': np.min(tool_call_counts),
        'tool_metrics/unique_tools/mean': np.mean(unique_tools_counts),
        'tool_metrics/usage_rate': np.mean(tool_usage_rates),
        'tool_metrics/think_usage_rate': np.mean(think_tag_usage),
        'tool_metrics/diversity/mean': np.mean(tool_diversities),
        'tool_metrics/most_used_tool': most_used_tool,
        'tool_metrics/total_unique_tools': len(set(all_used_tools)),
        
        # Invalid tool metrics
        'tool_metrics/invalid_call_count/mean': np.mean(invalid_tool_call_counts),
        'tool_metrics/invalid_call_count/total': sum(invalid_tool_call_counts),
        'tool_metrics/invalid_rate': np.mean([1.0 if count > 0 else 0.0 for count in invalid_tool_call_counts]),
        'tool_metrics/most_invalid_tool': most_invalid_tool,
        'tool_metrics/total_invalid_types': len(set(all_invalid_tools)),
        
        # Answer correctness metrics
        'tool_metrics/accuracy': np.mean(correctness_indicators),
        'tool_metrics/correct_count': sum(correctness_indicators),
        'tool_metrics/total_count': len(correctness_indicators),
    }
    
    # Add individual tool call counts
    for tool_name in valid_tool_names:
        count = batch_tool_counter.get(tool_name, 0)
        metrics[f'tool_metrics/individual/{tool_name}'] = count
    
    # Don't log individual invalid tools to avoid WandB clutter
    # Instead, just log aggregated statistics which are already included above
    
    # Update and add cumulative tracking
    global _CUMULATIVE_INVALID_TOOLS, _CUMULATIVE_TOTAL_RESPONSES
    
    current_invalid_total = sum(invalid_tool_call_counts)
    current_response_total = len(correctness_indicators)
    
    _CUMULATIVE_INVALID_TOOLS += current_invalid_total
    _CUMULATIVE_TOTAL_RESPONSES += current_response_total
    
    # Add cumulative metrics
    metrics['tool_metrics/cumulative_invalid_tools'] = _CUMULATIVE_INVALID_TOOLS
    metrics['tool_metrics/cumulative_total_responses'] = _CUMULATIVE_TOTAL_RESPONSES
    if _CUMULATIVE_TOTAL_RESPONSES > 0:
        metrics['tool_metrics/cumulative_invalid_rate'] = (
            _CUMULATIVE_INVALID_TOOLS / _CUMULATIVE_TOTAL_RESPONSES * 100
        )
    
    return metrics


def compute_data_metrics(batch: DataProto, use_critic: bool = True, tokenizer=None) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        
        # Add tool calling metrics
        **_compute_tool_metrics(batch, tokenizer),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        'perf/total_num_tokens': total_num_tokens,
        'perf/time_per_step': time,
        'perf/throughput': total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(data: list[Any],
                     subset_size: int,
                     reduce_fns: list[Callable[[np.ndarray], float]],
                     n_bootstrap: int = 1000,
                     seed: int = 42) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(data_sources: list[str],
                               sample_inputs: list[str],
                               infos_dict: dict[str, list[Any]],
                               seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.
    
    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infos_dict: variable name -> list of values for each sample
        
    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)
                metric[f"std@{n_resps}"] = np.std(var_vals)

                ns = []
                n = 2
                while n < n_resps:
                    ns.append(n)
                    n *= 2
                ns.append(n_resps)

                for n in ns:
                    # Best/Worst-of-N
                    [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals,
                                                                                  subset_size=n,
                                                                                  reduce_fns=[np.max, np.min],
                                                                                  seed=seed)
                    metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                    metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                    # Majority voting
                    if var2vals.get("pred", None) is not None:
                        vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                        [(maj_n_mean, maj_n_std)
                        ] = bootstrap_metric(data=vote_data,
                                             subset_size=n,
                                             reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                             seed=seed)
                        metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val
