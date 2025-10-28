mentor_template_sys = """In this environment you have access to a set of tools you can use to assist with the user query. \
You may perform multiple rounds of function calls. \
In each round, you can call one or more functions.

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

# Qwen3 native template - 시스템 프롬프트는 chat template이 자동 생성
qwen3_native_template = None  # Qwen3는 tools 파라미터로 처리

prompt_template_dict = {}
prompt_template_dict['mentor_template_sys'] = mentor_template_sys
prompt_template_dict['qwen3_native'] = qwen3_native_template
