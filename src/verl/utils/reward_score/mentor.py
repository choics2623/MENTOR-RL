"""
MENTOR Reward Function - Tool Exact Matching
=============================================
Main MENTOR reward function for tool-augmented mathematical reasoning:
- Base: 0.7 for correctness + 0.3 for exact tool matching + 0.1 for no hallucination
- Max total: 1.1 (0.7 + 0.3 + 0.1)
- Focuses on correctness with tool matching bonus and hallucination penalty
"""

import re
import json
import string
from typing import Union, List, Dict, Any
from collections import Counter

# Math-Verify imports for unified judge
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from math_verify import parse as mv_parse, verify as mv_verify

# Import mathruler grader
try:
    from mathruler.grader import grade_answer
    MATHRULER_AVAILABLE = True
except ImportError:
    MATHRULER_AVAILABLE = False

def remove_boxed(s):
    """Extract content from \\boxed{} - reused from existing code"""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"
    assert s[:len(left)] == left
    assert s[-1] == "}"
    return s[len(left):-1]

def last_boxed_only_string(string):
    """Find the last \\boxed{} in string - reused from existing code"""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    return retval

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string
    
def normalize_math_expression(s):
    """Normalize mathematical expressions for better comparison"""
    # linebreaks
    s = s.replace("\n", "")
    
    # remove inverse spaces
    s = s.replace("\\!", "")
    
    # replace \\ with \
    s = s.replace("\\\\", "\\")
    
    # replace tfrac and dfrac with frac
    s = s.replace("tfrac", "frac")
    s = s.replace("dfrac", "frac")
    
    # remove dollar signs
    s = s.replace("\\$", "")
    s = s.replace("$", "")
    
    # remove percentage
    s = s.replace("\\%", "")
    s = s.replace("%", "")
    
    # remove units (on the right)
    string = remove_right_units(string)
    return s

def normalize_answer(s):
    """Normalize answer for comparison - reused from existing code"""
    def remove_articles(text):
        return re.sub(r"\\b(a|an|the)\\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]):
    """Calculate F1 score - reused from existing code"""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

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

# Math-Verify metric instance
_mv_metric = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)

def mv_metric_score(pred: str, gold: Union[str, List[str]]) -> float:
    """Calculate Math-Verify metric score"""
    golds = [gold] if isinstance(gold, str) else gold
    best = 0.0
    for g in golds:
        try:
            score, _ = _mv_metric([f"\\boxed{{{g}}}"], [pred])
            best = max(best, float(score))
        except Exception:
            pass
    return best

def mv_parse_equal(pred: str, gold: Union[str, List[str]]) -> bool:
    """Check Math-Verify parse/verify equality"""
    golds = [gold] if isinstance(gold, str) else gold
    try:
        ans_parsed = mv_parse(pred)
        for g in golds:
            if mv_verify(mv_parse(g), ans_parsed):
                return True
    except Exception:
        pass
    return False

def mathruler_grade(prediction: str, ground_truth: Union[str, List[str]]) -> bool:
    """Check if prediction equals ground truth using mathruler grader"""
    if not MATHRULER_AVAILABLE:
        return False
        
    golds = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    try:
        for gt in golds:
            # Use mathruler's grade_answer function directly
            if grade_answer(prediction, gt):
                return True
    except Exception:
        pass
    return False

def unified_judge(prediction: str, ground_truth: Union[str, List[str]]) -> bool:
    """Unified judge using F1, MV metric, and MV parse/verify"""
    # Clean ground truth: remove boxed if present
    if isinstance(ground_truth, list):
        clean_ground_truth = []
        for gt in ground_truth:
            if '\\boxed{' in gt:
                clean_gt = remove_boxed(last_boxed_only_string(gt))
            else:
                clean_gt = gt
            # Apply math expression normalization
            clean_ground_truth.append(clean_gt)
    else:
        if '\\boxed{' in ground_truth:
            clean_ground_truth = remove_boxed(last_boxed_only_string(ground_truth))
        else:
            clean_ground_truth = ground_truth
        # Apply math expression normalization
        clean_ground_truth = clean_ground_truth
    
    # Normalize prediction as well
    normalized_prediction = prediction
    
    # Method 1: F1 normalization (commented out)
    # f1 = get_f1_score(normalized_prediction, clean_ground_truth)
    
    # Method 2: Math-Verify metric (uses normalized versions)
    mv_s = mv_metric_score(normalized_prediction, clean_ground_truth)
    
    # Method 3: Math-Verify parse/verify
    mv_b = mv_parse_equal(normalized_prediction, clean_ground_truth)
    
    # Method 4: Mathruler grader (uses original prediction and ground truth)
    mr_g = mathruler_grade(prediction, ground_truth)

    print(f"[DEBUG] ground_truth={ground_truth}, clean_ground_truth={clean_ground_truth}, prediction={prediction}, normalized_pred={normalized_prediction}, mv_s={mv_s}, mv_b={mv_b}, mr_g={mr_g}, correct?={(mv_s > 0) or mv_b or mr_g}")            
    # Return True if ANY method passes
    return (mv_s > 0) or mv_b or mr_g

def extract_tool_usage(solution_str: str) -> Dict[str, Any]:
    """Extract tool usage information from solution - reused from T1"""
    valid_tool_names = {
        'add', 'subtract', 'multiply', 'divide', 'sum_numbers',
        'floor', 'ceil', 'round_number', 'power', 'sqrt', 
        'abs_value', 'modulo'
    }
    
    tool_pattern = r'<tool_call>(.*?)</tool_call>'
    tool_calls = re.findall(tool_pattern, solution_str, re.DOTALL)
    
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
            name_match = re.search(r'"name":\\s*"([^"]+)"', tool_call)
            if name_match:
                tool_name = name_match.group(1)
        
        if tool_name:
            if tool_name not in valid_tool_names:
                invalid_tools.append(tool_name)
            else:
                used_tools.append(tool_name)
    
    return {
        'used_tools': used_tools,
        'invalid_tools': invalid_tools,
        'total_tool_calls': len(tool_calls)
    }

def compute_score_with_format(tokenizer, solution_str, ground_truth, teacher_tools=None) -> tuple[float, str]:
    """
    MENTOR reward function - Tool Exact Matching
    For datasets with tool usage only (teacher always uses tools)
    Base: 0.7 for correctness + 0.3 for exact tool matching + 0.1 for no hallucination
    Max total: 1.1 (0.7 + 0.3 + 0.1)
    """
    
    # 1. Check EOS token
    if not solution_str.endswith(tokenizer.eos_token):
        return 0.0, '[MENTOR] Missing EOS token'
    
    # 2. Extract boxed answer
    try:
        boxed_string = last_boxed_only_string(solution_str)
        if not boxed_string:
            return 0.1, '[MENTOR] Missing boxed answer but valid format'
        
        answer = remove_boxed(boxed_string)
        if not answer.strip():
            return 0.1, '[MENTOR] Empty boxed answer but valid format'
            
    except Exception as e:
        return 0.1, f'[MENTOR] Answer extraction error: {e}'
    
    # 3. Apply unified judge (scaled to 0.7)
    is_correct = unified_judge(answer, ground_truth)
    correctness_reward = 0.7 if is_correct else 0.0
    
    # 4. Tool usage analysis
    tool_bonus = 0.0
    bonus_reason = ""
    
    # Check if teacher used tools (problem requires tools)
    teacher_used_tools = teacher_tools is not None and len(teacher_tools) > 0
    teacher_tool_set = set(teacher_tools) if teacher_used_tools else set()
    
    # Check if student used tools
    student_tool_info = extract_tool_usage(solution_str)
    student_used_tools = len(student_tool_info['used_tools']) > 0
    student_tool_set = set(student_tool_info['used_tools'])
    
    # MENTOR Logic: Tool exact matching only (teacher always uses tools)
    tool_matching = 0.0

    # Teacher should always have tools in this dataset
    if not teacher_used_tools:
        return 0.1, '[MENTOR] Error: Teacher tools not provided'

    # Check if student used tools
    if student_used_tools:
        # MENTOR: Exact tool matching
        if teacher_tool_set == student_tool_set:
            tool_matching = 0.3
            bonus_reason = f"Exact match (+0.3) (T:{sorted(teacher_tool_set)}, S:{sorted(student_tool_set)})"
        else:
            bonus_reason = f"Tool mismatch (+0.0) (T:{sorted(teacher_tool_set)}, S:{sorted(student_tool_set)})"
    else:
        bonus_reason = f"Student didn't use tools (+0.0) (T:{sorted(teacher_tool_set)}, S:[])"

    # Add no-hallucination bonus (0.1 for no invalid tools)
    no_hallucination_bonus = 0.0
    if len(student_tool_info['invalid_tools']) == 0:
        no_hallucination_bonus = 0.1
        bonus_reason += " & no hallucination (+0.1)"
    elif len(student_tool_info['invalid_tools']) > 0:
        bonus_reason += f" but {len(student_tool_info['invalid_tools'])} invalid tools: {student_tool_info['invalid_tools']}"

    # Scale down tool bonus for wrong answers
    if is_correct:
        tool_bonus = tool_matching + no_hallucination_bonus
        final_reward = correctness_reward + tool_bonus
        print(f'[MENTOR] Correct answer, Correctness={correctness_reward:.1f}, {bonus_reason}')
        return final_reward, f'[MENTOR] Correct answer, Correctness={correctness_reward:.1f}, {bonus_reason}'
    else:
        # Wrong answer: very small bonus for tool matching (0.06 max) + full no-hallucination bonus
        tool_bonus_scaled = tool_matching * 0.2 + no_hallucination_bonus  # 0.3 * 0.2 + 0.1 = 0.16 max
        final_reward = correctness_reward + tool_bonus_scaled
        bonus_reason_scaled = bonus_reason.replace('+0.3', f'+{tool_matching * 0.2:.1f}')
        print(f'[MENTOR] Wrong answer, Correctness={correctness_reward:.1f}, {bonus_reason_scaled} (scaled down)')
        return final_reward, f'[MENTOR] Wrong answer, Correctness={correctness_reward:.1f}, {bonus_reason_scaled} (scaled down)'