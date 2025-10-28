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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import json
import numpy as np

class MentorRewardManagerWithSave():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, save_path=None, ablation_exp_id=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.save_path = save_path
        self.ablation_exp_id = ablation_exp_id
        
        # Statistics tracking
        self.total_invalid_tool_calls = 0
        self.total_responses = 0
        self.invalid_tool_names = {}  # track count of each invalid tool name

    def __call__(self, data: DataProto, return_dict=False, curr_save_path=None):
        """We will expand this function gradually based on the available datasets"""

        if curr_save_path is not None:
            save_path = curr_save_path
        else:
            save_path = self.save_path

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        if save_path is not None:
            save_file = open(save_path, 'a')

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            # Extract extra_info for ablation experiments
            extra_info = data_item.non_tensor_batch.get('extra_info', {})

            # Add tool_calls_used to extra_info if available
            if 'tool_calls_used' in data_item.non_tensor_batch:
                # Parse extra_info if it's a string
                if isinstance(extra_info, str):
                    try:
                        extra_info = json.loads(extra_info)
                    except:
                        extra_info = {}
                elif not isinstance(extra_info, dict):
                    extra_info = {}
                else:
                    extra_info = dict(extra_info)  # Make a copy

                # Parse tool_calls_used if it's a JSON string
                tool_calls_used = data_item.non_tensor_batch['tool_calls_used']

                if isinstance(tool_calls_used, str):
                    try:
                        tool_calls_used = json.loads(tool_calls_used)
                    except json.JSONDecodeError:
                        tool_calls_used = []
                elif isinstance(tool_calls_used, np.ndarray):
                    tool_calls_used = tool_calls_used.tolist()

                extra_info['tool_calls_used'] = tool_calls_used
            
            score = self.compute_score(
                data_source=data_source,
                tokenizer=self.tokenizer,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                ablation_exp_id=self.ablation_exp_id,
            )
            if isinstance(score, tuple):
                score, reason = score
            else:
                reason = ''
            reward_tensor[i, valid_response_length - 1] = score

            # Track invalid tool call statistics for big_math and mathmedium2
            tool_info = None
            num_invalid = 0
            if data_source in ['nvidia/AceReason-Math']:
                self.total_responses += 1
                # Extract tool usage info from solution
                # Use mentor_ablation2 as reference (all QWEN3 files have same extract_tool_usage)
                from verl.utils.reward_score.mentor_ablation2 import extract_tool_usage
                tool_info = extract_tool_usage(sequences_str)

                # Calculate has_invalid and num_invalid from tool_info
                num_invalid = len(tool_info['invalid_tools'])
                has_invalid = num_invalid > 0

                if has_invalid:
                    self.total_invalid_tool_calls += num_invalid
                    # Track individual invalid tool names
                    for invalid_tool in tool_info['invalid_tools']:
                        if invalid_tool not in self.invalid_tool_names:
                            self.invalid_tool_names[invalid_tool] = 0
                        self.invalid_tool_names[invalid_tool] += 1

            if save_path is not None:
                save_json_line = {
                    'data_source': data_source,
                    'sequences_str': sequences_str,
                    'ground_truth': ground_truth,
                    'score': score,
                    'reason': reason
                }
                # Add invalid tool call info for tracking (only count, not individual names to reduce log size)
                if data_source in ['nvidia/AceReason-Math']:
                    save_json_line['num_invalid_tools'] = num_invalid
                    # Only save individual tool names if there are invalid tools and it's a small number
                    if num_invalid > 0 and num_invalid <= 3:
                        save_json_line['invalid_tool_calls'] = tool_info['invalid_tools']
                save_file.write(json.dumps(save_json_line, ensure_ascii=False) + '\n')

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print('-' * 20)
                print(f"data_source: \n{data_source}")
                print(f"sequences_str: \n{sequences_str}")
                print(f"ground_truth: \n{ground_truth}")
                print(f"score: \n{score}")  
                print(f"reason: \n{reason}")
                print('-' * 20)

        if save_path is not None:
            save_file.close()

        # Print invalid tool call statistics
        if self.total_responses > 0:
            invalid_rate = (self.total_invalid_tool_calls / self.total_responses) * 100
            print(f"\n📊 Invalid Tool Call Statistics:")
            print(f"Total responses processed: {self.total_responses}")
            print(f"Total invalid tool calls: {self.total_invalid_tool_calls}")
            print(f"Invalid tool call rate: {invalid_rate:.2f}%")
            
            if self.invalid_tool_names:
                print(f"Most common invalid tools:")
                sorted_invalid = sorted(self.invalid_tool_names.items(), key=lambda x: x[1], reverse=True)
                for tool_name, count in sorted_invalid[:5]:  # Top 5
                    print(f"  - {tool_name}: {count} times")
            print()

        if return_dict:
            return_data = {
                "reward_tensor": reward_tensor,
            }
            # Add invalid tool call statistics to return dict for logging (aggregated only)
            if self.total_responses > 0:
                return_data["invalid_tool_stats"] = {
                    "total_invalid_calls": self.total_invalid_tool_calls,
                    "total_responses": self.total_responses,
                    "invalid_rate": (self.total_invalid_tool_calls / self.total_responses) * 100,
                    # Only include top 3 most common invalid tools to reduce noise
                    "top_invalid_tools_count": len(self.invalid_tool_names),
                    "most_common_invalid": dict(sorted(self.invalid_tool_names.items(), key=lambda x: x[1], reverse=True)[:3]) if self.invalid_tool_names else {}
                }
            return return_data
        else:
            return reward_tensor