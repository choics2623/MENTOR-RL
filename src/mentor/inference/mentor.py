import re
import json
import requests
import time
from typing import List
from functools import wraps

def retry(max: int=10, sleep: int=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_e = None
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_e = e
                    print(f"[retry] try {i} times: {e}")
                    if sleep and i < max - 1:
                        time.sleep(sleep)
            return {"_retry_failed": True, "error": str(last_e)}
        return wrapper
    return decorator

class Mentor():
    """"If the tokenizer does not contain a chat template, the custom system prompt from [ReCall](https://github.com/Agent-RL/ReCall) is used. By default, the implementation follows the Qwen chat template."""""
    
    system_prompt = """In this environment you have access to a set of tools you can use to assist with the user query. \
You may perform multiple rounds of function calls. \
In each round, you can call one or more functions. \

Here are available functions in JSONSchema format: \n```json\n{func_schemas}\n```

In your response, you need to first think about the reasoning process in the mind and then conduct function calling to get the information or perform the actions if needed. \
The reasoning process and function calling are enclosed within <think> </think> and <tool_call> </tool_call> tags. \
The results of the function calls will be given back to you after execution, \
and you can continue to call functions until you get the final answer for the user's question. \
Finally, if you have got the answer, enclose it within \\boxed{{}} with latex format and do not continue to call functions, \
i.e., <think> Based on the response from the function call, I get the weather information. </think> The weather in Beijing on 2025-04-01 is \\[ \\boxed{{20C}} \\].

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

    def __init__(self, model_url, executor_url, model_name='qwen3'):
        self.model_url = model_url
        self.executor_url = executor_url
        self.model_name = model_name

        # Initialize tool calls tracking
        self.tool_calls_history = []
        self.tool_calls_count = 0
        self.successful_tool_calls = 0
        self.failed_tool_calls = 0

    def reset_tool_calls_history(self):
        """Reset tool calls history for a new inference run"""
        self.tool_calls_history = []
        self.tool_calls_count = 0
        self.successful_tool_calls = 0
        self.failed_tool_calls = 0

    def init_prompt(self, func_schemas, question):
        system_prompt = f"<|im_start|>system\n{self.system_prompt.format(func_schemas=func_schemas)}<|im_end|>"
        user_prompt = f"<|im_start|>user\n{question}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return system_prompt + "\n" + user_prompt + "\n" + assistant_prefix

    def cat_assistant_response(self, curr_prompt, assistant_response):
        return curr_prompt + assistant_response + "<|im_end|>"
    
    def cat_tool_results(self, curr_prompt, tool_calls, results):
        tool_response_str = ""
        for tool_call, result in zip(tool_calls, results):
            tool_response_str += f"<tool_response>{tool_call}\n{result}\n</tool_response>\n"
        tool_response_str = f"<|im_start|>user\n{tool_response_str}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return curr_prompt + "\n" + tool_response_str + "\n" + assistant_prefix

    def format_tool_call(self, tool_call_str: str):
        """Convert JSON function call description to Python executable code string."""
        try:
            call_json = json.loads(tool_call_str)
            func_name = call_json['name']
            arguments = call_json.get('arguments', {})
            
            args_str = ', '.join(f"{k}={repr(v)}" for k, v in arguments.items())
            return f"{func_name}({args_str})"
        except Exception as e:
            return f"Parse tool call failed: {e}"
    
    def execute_tool_calls(self, env: str, tool_calls: List[str]) -> List[str]:
        def exe_tool_call(env, call):
            url = self.executor_url + '/execute'

            call_str = self.format_tool_call(call)
            if call_str.startswith("error: parse tool call failed"):
                return call_str

            try:
                data = {
                    'env': env,
                    'call': call_str
                }
                response = requests.post(url, json=data, timeout=180)
                if response.status_code != 200:
                    return f"error: {response.status_code}"
                response = response.json()
                ret_str = ''
                if response['result']:
                    ret_str += f'result: \n{response["result"]}\n'
                if response['output']:
                    ret_str += f'output: \n{response["output"]}\n'
                if response['error']:
                    ret_str += f'error: \n{response["error"]}\n'
                return ret_str.strip()
            except requests.exceptions.Timeout:
                return "error: execution timed out"
            except Exception as e:
                return str(e)
        
        results = []
        for tool_call in tool_calls:
            result = exe_tool_call(env, tool_call)
            results.append(result)
        return results
    
    def validate_tool_calls(self, output_str):
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

    def extract_tool_calls(self, output_str):
        if not self.validate_tool_calls(output_str):
            return []

        try:
            pattern = r'<tool_call>((?:(?!</tool_call>).)*)</tool_call>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []

    def _parse_qwen3_response(self, text):
        # extract reasoning_content (<think> tag content)
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, text, re.DOTALL)
        reasoning_content = think_match.group(1).strip() if think_match else None
        
        # extract tool_calls (<tool_call> tag content)
        tool_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_matches = re.finditer(tool_pattern, text, re.DOTALL)
        tool_calls = []
        for match in tool_matches:
            try:
                # try to parse JSON content
                tool_content = match.group(1).strip()
                tool_json = json.loads(tool_content)
                assert 'name' in tool_json and 'arguments' in tool_json, f"name or arguments not in tool_json: {tool_json}"
                assert isinstance(tool_json['arguments'], dict), f"arguments is not a dict: {tool_json['arguments']}"
                tool_calls.append(tool_content)
            except Exception as e:
                # if JSON parsing fails, add the original string
                # tool_calls.append(tool_content)
                tool_calls.append('error: parse tool call failed')
        
        # remove all tags, get the remaining text
        clean_text = text
        clean_text = re.sub(think_pattern, '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(tool_pattern, '', clean_text, flags=re.DOTALL)
        content = clean_text.strip()
        
        return {
            'reasoning_content': reasoning_content,
            'tool_calls': tool_calls,
            'content': content
        }
    
    def _pick_text(self, payload):
        if not isinstance(payload, dict):
            return None
        if 'text' in payload and isinstance(payload['text'], str):
            return payload['text']
        # vLLM / 일부 서버 호환
        if 'outputs' in payload and payload['outputs']:
            first = payload['outputs'][0]
            if isinstance(first, dict) and 'text' in first:
                return first['text']
        if 'output_text' in payload and isinstance(payload['output_text'], str):
            return payload['output_text']
        return None

    def _pick_text_openai_format(self, payload):
        """Extract text from OpenAI compatible API response"""
        if not isinstance(payload, dict):
            return None
        if 'choices' in payload and payload['choices']:
            first_choice = payload['choices'][0]
            if isinstance(first_choice, dict) and 'text' in first_choice:
                return first_choice['text']
        return None

    @retry(max=5, sleep=1)
    def run(self, env, func_schemas, question):
        curr_prompt = self.init_prompt(func_schemas, question)
        for _ in range(5):
            try:
                resp = requests.post(
                    f'{self.model_url}/generate',
                    json={
                        "text": curr_prompt,
                        "sampling_params": {"temperature": 0.0, "max_new_tokens": 512}
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                payload = resp.json()
            except Exception as e:
                print(f"LLM request failed: {e}")
                # retry 데코레이터가 다음 시도로 넘김
                raise

            print(f"LLM response: {payload}")
            text = self._pick_text(payload)
            if not text:
                # text가 없으면 재시도
                raise ValueError(f"LLM response missing 'text': {payload}")

            curr_prompt = self.cat_assistant_response(curr_prompt, text)

            if self.model_name == 'qwen3':
                parsed_response = self._parse_qwen3_response(text)
                tool_calls = parsed_response['tool_calls']
            else:
                tool_calls = self.extract_tool_calls(text)

            if len(tool_calls) == 0:
                break

            results = self.execute_tool_calls(env, tool_calls)
            curr_prompt = self.cat_tool_results(curr_prompt, tool_calls, results)

        return curr_prompt

    def run_qwen3_native(self, prompt, env, tools):
        """Run Mentor with Qwen3 native tool calling format"""
        # Reset tool calls history for this run
        self.reset_tool_calls_history()

        curr_prompt = prompt

        for turn_idx in range(5):
            try:
                # Use OpenAI compatible API format for SGLang
                resp = requests.post(
                    f'{self.model_url}/v1/completions',
                    json={
                        'prompt': curr_prompt,
                        'max_tokens': 12000,
                        'temperature': 0.6,
                        'top_p': 0.95,
                        'top_k': 20,
                        'min_p': 0,
                        'stop': ['<|im_end|>'],
                    },
                    headers={'Content-Type': 'application/json'},
                    timeout=180,
                )
                resp.raise_for_status()
                payload = resp.json()
            except Exception as e:
                print(f"LLM request failed: {e}")
                raise

            print(f"LLM response: {payload}")
            text = self._pick_text_openai_format(payload)
            if not text:
                # Fallback to original method
                text = self._pick_text(payload)
            if not text:
                raise ValueError(f"LLM response missing 'text': {payload}")

            curr_prompt = self.cat_assistant_response(curr_prompt, text)

            # Parse Qwen3 native tool calls (if any)
            if self.model_name == 'qwen3':
                parsed_response = self._parse_qwen3_response(text)
                tool_calls = parsed_response['tool_calls']
            else:
                tool_calls = self.extract_tool_calls(text)

            if len(tool_calls) == 0:
                break

            results = self.execute_tool_calls(env, tool_calls)

            # Record each tool call with its result
            # Count both total and success/failure in the same loop for consistency
            for tool_call, result in zip(tool_calls, results):
                # Increment total count
                self.tool_calls_count += 1

                # Check success/failure
                is_success = not (isinstance(result, str) and result.startswith("error:"))
                if is_success:
                    self.successful_tool_calls += 1
                else:
                    self.failed_tool_calls += 1

                self.tool_calls_history.append({
                    'turn': turn_idx,
                    'tool_call': tool_call,
                    'result': result,
                    'success': is_success
                })

            curr_prompt = self.cat_tool_results(curr_prompt, tool_calls, results)

        return curr_prompt