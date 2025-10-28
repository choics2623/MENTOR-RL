"""
SPARSE Reward Function
=================================
Simplified reward for Qwen3 native tool calling:
- Only checks EOS token, boxed answer, and F1 score
- No template validation (handled by Qwen3 chat template)
- No <think> tag validation (handled by Qwen3 native thinking)
- No tool call format validation (handled by Qwen3 native tool calling)
"""

import re
import string
from typing import Union, List
from collections import Counter

# Math-Verify imports for unified judge
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from math_verify import parse as mv_parse, verify as mv_verify

# Import mathruler grader
try:
    from mathruler.grader import extract_boxed_content, grade_answer
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
        return re.sub(r"\b(a|an|the)\b", " ", text)

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

def compute_score_with_format(tokenizer, solution_str, ground_truth, teacher_tools=None) -> tuple[float, str]:
    """
    SPARSE reward function with Unified Judge
    Only checks: EOS token + boxed answer + Unified Judge (F1 + MV-Metric + MV-Verify)
    No tool call validation needed - rollout handles errors automatically
    """
    
    # 1. Check EOS token
    if not solution_str.endswith(tokenizer.eos_token):
        print("""0.0, '[SPARSE] Missing EOS token'""")
        return 0.0, '[SPARSE] Missing EOS token'
    
    # 2. Extract boxed answer
    try:
        boxed_string = last_boxed_only_string(solution_str)
        if not boxed_string:
            print("""0.1, '[SPARSE] Missing boxed answer but valid format'""")
            return 0.1, '[SPARSE] Missing boxed answer but valid format'
        
        answer = remove_boxed(boxed_string)
        if not answer.strip():
            print("""0.1, '[SPARSE] Empty boxed answer but valid format'""")
            return 0.1, '[SPARSE] Empty boxed answer but valid format'

    except Exception as e:
        print(f"""0.1, '[SPARSE] Answer extraction error: {e}'""")
        return 0.1, f'[SPARSE] Answer extraction error: {e}'
    
    # 3. Apply unified judge (ANY of F1/MV-Metric/MV-Verify passes)
    is_correct = unified_judge(answer, ground_truth)
    
    

    if is_correct:
        print(f"""1.0, f'[SPARSE] Correct answer by unified judge, predicted="{answer}""")
        return 1.0, f'[SPARSE] Correct answer by unified judge, predicted="{answer}"'
    else:
        print(f"""0.1, f'[SPARSE] Wrong answer but good format, predicted="{answer}""")
        return 0.1, f'[SPARSE] Wrong answer but good format, predicted="{answer}"'
    