from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

import structlog

logger = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _PROJECT_ROOT / "data"

TRAINING_BUGS_DIR = str(_DATA_ROOT / "training_bugs")
TRAINING_CLEAN_DIR = str(_DATA_ROOT / "training_clean")
HELD_OUT_BUGS_DIR = str(_DATA_ROOT / "held_out_bugs")
HELD_OUT_CLEAN_DIR = str(_DATA_ROOT / "held_out_clean")
VALIDATION_DIR = str(_DATA_ROOT / "validation")
GROUND_TRUTH_PATH = str(_DATA_ROOT / "ground_truth.json")


class _FileSpec(NamedTuple):
    filename: str
    bug_code: str
    clean_code: str
    bug_type: str
    bug_line: int
    bug_description: str
    detectable_by_pylint: bool
    detectable_by_ast: bool
    detectable_by_execution: bool
    split: str


_TRAINING_SPECS: list[_FileSpec] = [
    # ── undefined_variable (6 files) ──────────────────────────────────────────
    _FileSpec(
        "train_undef_001.py",
        """\
def compute_total(prices: list[float]) -> float:
    total = 0.0
    for price in prices:
        total += price * tax_rate
    return total
""",
        """\
TAX_RATE = 0.08

def compute_total(prices: list[float]) -> float:
    total = 0.0
    for price in prices:
        total += price * TAX_RATE
    return total
""",
        "undefined_variable", 4,
        "Variable 'tax_rate' used before definition",
        True, False, True, "training",
    ),
    _FileSpec(
        "train_undef_002.py",
        """\
def format_greeting(name: str) -> str:
    return f"{prefix} {name}!"
""",
        """\
PREFIX = "Hello"

def format_greeting(name: str) -> str:
    return f"{PREFIX} {name}!"
""",
        "undefined_variable", 2,
        "Variable 'prefix' used before definition",
        True, False, True, "training",
    ),
    _FileSpec(
        "train_undef_003.py",
        """\
def scale_vector(values: list[float]) -> list[float]:
    return [v * scale_factor for v in values]
""",
        """\
SCALE_FACTOR = 2.0

def scale_vector(values: list[float]) -> list[float]:
    return [v * SCALE_FACTOR for v in values]
""",
        "undefined_variable", 2,
        "Variable 'scale_factor' used before definition",
        True, False, True, "training",
    ),
    _FileSpec(
        "train_undef_004.py",
        """\
def apply_discount(price: float) -> float:
    discounted = price - (price * discount_pct)
    return round(discounted, 2)
""",
        """\
DISCOUNT_PCT = 0.10

def apply_discount(price: float) -> float:
    discounted = price - (price * DISCOUNT_PCT)
    return round(discounted, 2)
""",
        "undefined_variable", 2,
        "Variable 'discount_pct' used before definition",
        True, False, True, "training",
    ),
    _FileSpec(
        "train_undef_005.py",
        """\
def repeat_string(text: str) -> str:
    return text * repeat_count
""",
        """\
REPEAT_COUNT = 3

def repeat_string(text: str) -> str:
    return text * REPEAT_COUNT
""",
        "undefined_variable", 2,
        "Variable 'repeat_count' used before definition",
        True, False, True, "training",
    ),
    _FileSpec(
        "train_undef_006.py",
        """\
def build_url(path: str) -> str:
    return base_url + path
""",
        """\
BASE_URL = "https://example.com"

def build_url(path: str) -> str:
    return BASE_URL + path
""",
        "undefined_variable", 2,
        "Variable 'base_url' used before definition",
        True, False, True, "training",
    ),

    # ── mutable_default_argument (6 files) ─────────────────────────────────────
    _FileSpec(
        "train_mutable_001.py",
        """\
def collect_items(item: str, bucket: list[str] = []) -> list[str]:
    bucket.append(item)
    return bucket
""",
        """\
def collect_items(item: str, bucket: list[str] | None = None) -> list[str]:
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
""",
        "mutable_default_argument", 1,
        "Mutable default argument [] shared across calls",
        True, True, False, "training",
    ),
    _FileSpec(
        "train_mutable_002.py",
        """\
def register_tag(tag: str, tags: list[str] = []) -> list[str]:
    if tag not in tags:
        tags.append(tag)
    return tags
""",
        """\
def register_tag(tag: str, tags: list[str] | None = None) -> list[str]:
    if tags is None:
        tags = []
    if tag not in tags:
        tags.append(tag)
    return tags
""",
        "mutable_default_argument", 1,
        "Mutable default argument [] shared across calls",
        True, True, False, "training",
    ),
    _FileSpec(
        "train_mutable_003.py",
        """\
def push_event(event: dict, log: list[dict] = []) -> list[dict]:
    log.append(event)
    return log
""",
        """\
def push_event(event: dict, log: list[dict] | None = None) -> list[dict]:
    if log is None:
        log = []
    log.append(event)
    return log
""",
        "mutable_default_argument", 1,
        "Mutable default argument [] shared across calls",
        True, True, False, "training",
    ),
    _FileSpec(
        "train_mutable_004.py",
        """\
def add_score(score: int, scores: list[int] = []) -> list[int]:
    scores.append(score)
    return sorted(scores)
""",
        """\
def add_score(score: int, scores: list[int] | None = None) -> list[int]:
    if scores is None:
        scores = []
    scores.append(score)
    return sorted(scores)
""",
        "mutable_default_argument", 1,
        "Mutable default argument [] shared across calls",
        True, True, False, "training",
    ),
    _FileSpec(
        "train_mutable_005.py",
        """\
def merge_options(key: str, value: str, opts: dict[str, str] = {}) -> dict[str, str]:
    opts[key] = value
    return opts
""",
        """\
def merge_options(key: str, value: str, opts: dict[str, str] | None = None) -> dict[str, str]:
    if opts is None:
        opts = {}
    opts[key] = value
    return opts
""",
        "mutable_default_argument", 1,
        "Mutable default argument {} shared across calls",
        True, True, False, "training",
    ),
    _FileSpec(
        "train_mutable_006.py",
        """\
def track_visit(page: str, history: list[str] = []) -> list[str]:
    history.append(page)
    return history
""",
        """\
def track_visit(page: str, history: list[str] | None = None) -> list[str]:
    if history is None:
        history = []
    history.append(page)
    return history
""",
        "mutable_default_argument", 1,
        "Mutable default argument [] shared across calls",
        True, True, False, "training",
    ),

    # ── bare_except (6 files) ──────────────────────────────────────────────────
    _FileSpec(
        "train_bare_001.py",
        """\
def safe_parse_int(text: str) -> int | None:
    try:
        return int(text)
    except:
        return None
""",
        """\
def safe_parse_int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None
""",
        "bare_except", 4,
        "Bare except catches all exceptions including KeyboardInterrupt",
        True, True, False, "training",
    ),
    _FileSpec(
        "train_bare_002.py",
        """\
def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            import json
            return json.load(f)
    except:
        return {}
""",
        """\
import json

def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
""",
        "bare_except", 5,
        "Bare except catches all exceptions including KeyboardInterrupt",
        True, True, False, "training",
    ),
    _FileSpec(
        "train_bare_003.py",
        """\
def fetch_value(data: dict, key: str) -> object:
    try:
        return data[key]
    except:
        return None
""",
        """\
def fetch_value(data: dict, key: str) -> object:
    try:
        return data[key]
    except KeyError:
        return None
""",
        "bare_except", 4,
        "Bare except catches all exceptions including KeyboardInterrupt",
        True, True, False, "training",
    ),
    _FileSpec(
        "train_bare_004.py",
        """\
def convert_float(value: str) -> float:
    try:
        return float(value)
    except:
        return 0.0
""",
        """\
def convert_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0
""",
        "bare_except", 4,
        "Bare except catches all exceptions including KeyboardInterrupt",
        True, True, False, "training",
    ),
    _FileSpec(
        "train_bare_005.py",
        """\
def read_line(filename: str, lineno: int) -> str:
    try:
        with open(filename) as f:
            lines = f.readlines()
        return lines[lineno]
    except:
        return ""
""",
        """\
def read_line(filename: str, lineno: int) -> str:
    try:
        with open(filename) as f:
            lines = f.readlines()
        return lines[lineno]
    except (OSError, IndexError):
        return ""
""",
        "bare_except", 6,
        "Bare except catches all exceptions including KeyboardInterrupt",
        True, True, False, "training",
    ),
    _FileSpec(
        "train_bare_006.py",
        """\
def divide(a: float, b: float) -> float:
    try:
        return a / b
    except:
        return 0.0
""",
        """\
def divide(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        return 0.0
""",
        "bare_except", 4,
        "Bare except catches all exceptions including KeyboardInterrupt",
        True, True, False, "training",
    ),

    # ── equality_none_check (6 files) ──────────────────────────────────────────
    _FileSpec(
        "train_none_001.py",
        """\
def process_result(value: object) -> str:
    if value == None:
        return "empty"
    return str(value)
""",
        """\
def process_result(value: object) -> str:
    if value is None:
        return "empty"
    return str(value)
""",
        "equality_none_check", 2,
        "Use 'is None' not '== None' for None comparison",
        True, False, False, "training",
    ),
    _FileSpec(
        "train_none_002.py",
        """\
def get_label(item: dict) -> str:
    if item.get("label") == None:
        return "unlabeled"
    return item["label"]
""",
        """\
def get_label(item: dict) -> str:
    if item.get("label") is None:
        return "unlabeled"
    return item["label"]
""",
        "equality_none_check", 2,
        "Use 'is None' not '== None' for None comparison",
        True, False, False, "training",
    ),
    _FileSpec(
        "train_none_003.py",
        """\
def describe_user(user: dict | None) -> str:
    if user == None:
        return "anonymous"
    return user.get("name", "unknown")
""",
        """\
def describe_user(user: dict | None) -> str:
    if user is None:
        return "anonymous"
    return user.get("name", "unknown")
""",
        "equality_none_check", 2,
        "Use 'is None' not '== None' for None comparison",
        True, False, False, "training",
    ),
    _FileSpec(
        "train_none_004.py",
        """\
def validate_token(token: str | None) -> bool:
    if token != None and len(token) > 0:
        return True
    return False
""",
        """\
def validate_token(token: str | None) -> bool:
    if token is not None and len(token) > 0:
        return True
    return False
""",
        "equality_none_check", 2,
        "Use 'is not None' not '!= None' for None comparison",
        True, False, False, "training",
    ),
    _FileSpec(
        "train_none_005.py",
        """\
def safe_upper(text: str | None) -> str:
    if text == None:
        return ""
    return text.upper()
""",
        """\
def safe_upper(text: str | None) -> str:
    if text is None:
        return ""
    return text.upper()
""",
        "equality_none_check", 2,
        "Use 'is None' not '== None' for None comparison",
        True, False, False, "training",
    ),
    _FileSpec(
        "train_none_006.py",
        """\
def first_non_null(items: list[object]) -> object:
    for item in items:
        if item != None:
            return item
    return None
""",
        """\
def first_non_null(items: list[object]) -> object:
    for item in items:
        if item is not None:
            return item
    return None
""",
        "equality_none_check", 3,
        "Use 'is not None' not '!= None' for None comparison",
        True, False, False, "training",
    ),

    # ── shadowed_builtin (6 files) ─────────────────────────────────────────────
    _FileSpec(
        "train_shadow_001.py",
        """\
def count_items(list: list[int]) -> int:
    return len(list)
""",
        """\
def count_items(items: list[int]) -> int:
    return len(items)
""",
        "shadowed_builtin", 1,
        "Parameter 'list' shadows built-in",
        True, False, False, "training",
    ),
    _FileSpec(
        "train_shadow_002.py",
        """\
def serialize(input: object) -> str:
    return str(input)
""",
        """\
def serialize(value: object) -> str:
    return str(value)
""",
        "shadowed_builtin", 1,
        "Parameter 'input' shadows built-in",
        True, False, False, "training",
    ),
    _FileSpec(
        "train_shadow_003.py",
        """\
def categorize(type: str, items: list[str]) -> dict[str, list[str]]:
    return {type: items}
""",
        """\
def categorize(category: str, items: list[str]) -> dict[str, list[str]]:
    return {category: items}
""",
        "shadowed_builtin", 1,
        "Parameter 'type' shadows built-in",
        True, False, False, "training",
    ),
    _FileSpec(
        "train_shadow_004.py",
        """\
def wrap_value(id: int, value: str) -> dict[str, object]:
    return {"id": id, "value": value}
""",
        """\
def wrap_value(item_id: int, value: str) -> dict[str, object]:
    return {"id": item_id, "value": value}
""",
        "shadowed_builtin", 1,
        "Parameter 'id' shadows built-in",
        True, False, False, "training",
    ),
    _FileSpec(
        "train_shadow_005.py",
        """\
def to_string(format: str, value: float) -> str:
    return format % value
""",
        """\
def to_string(fmt: str, value: float) -> str:
    return fmt % value
""",
        "shadowed_builtin", 1,
        "Parameter 'format' shadows built-in",
        True, False, False, "training",
    ),
    _FileSpec(
        "train_shadow_006.py",
        """\
def get_bytes(str: str) -> bytes:
    return str.encode("utf-8")
""",
        """\
def get_bytes(text: str) -> bytes:
    return text.encode("utf-8")
""",
        "shadowed_builtin", 1,
        "Parameter 'str' shadows built-in",
        True, False, False, "training",
    ),
]

_HELD_OUT_SPECS: list[_FileSpec] = [
    # ── off_by_one (5 files) ────────────────────────────────────────────────────
    _FileSpec(
        "ood_obo_001.py",
        """\
def find_last(items: list[int], target: int) -> int:
    for i in range(len(items) - 1):
        if items[i] == target:
            return i
    return -1
""",
        """\
def find_last(items: list[int], target: int) -> int:
    for i in range(len(items)):
        if items[i] == target:
            return i
    return -1
""",
        "off_by_one", 2,
        "range(len(items) - 1) skips last element",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_obo_002.py",
        """\
def slice_middle(items: list[str]) -> list[str]:
    return items[1:len(items) - 1]
""",
        """\
def slice_middle(items: list[str]) -> list[str]:
    return items[1:len(items)]
""",
        "off_by_one", 2,
        "Off-by-one in slice end index discards last element",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_obo_003.py",
        """\
def count_pairs(n: int) -> int:
    count = 0
    for i in range(n - 1):
        for j in range(i + 1, n - 1):
            count += 1
    return count
""",
        """\
def count_pairs(n: int) -> int:
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            count += 1
    return count
""",
        "off_by_one", 3,
        "Inner range(n - 1) off-by-one prevents last element from pairing",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_obo_004.py",
        """\
def get_neighbors(grid: list[list[int]], r: int, c: int) -> list[int]:
    neighbors = []
    for dr in range(-1, 1):
        for dc in range(-1, 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                neighbors.append(grid[nr][nc])
    return neighbors
""",
        """\
def get_neighbors(grid: list[list[int]], r: int, c: int) -> list[int]:
    neighbors = []
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                neighbors.append(grid[nr][nc])
    return neighbors
""",
        "off_by_one", 3,
        "range(-1, 1) misses the +1 offset neighbor",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_obo_005.py",
        """\
def moving_average(data: list[float], window: int) -> list[float]:
    result = []
    for i in range(len(data) - window):
        window_sum = sum(data[i:i + window])
        result.append(window_sum / window)
    return result
""",
        """\
def moving_average(data: list[float], window: int) -> list[float]:
    result = []
    for i in range(len(data) - window + 1):
        window_sum = sum(data[i:i + window])
        result.append(window_sum / window)
    return result
""",
        "off_by_one", 3,
        "range(len(data) - window) off-by-one drops last window",
        False, False, False, "held_out",
    ),

    # ── wrong_return_variable (5 files) ─────────────────────────────────────────
    _FileSpec(
        "ood_wrv_001.py",
        """\
def swap(a: int, b: int) -> tuple[int, int]:
    temp = a
    a = b
    b = temp
    return a, a
""",
        """\
def swap(a: int, b: int) -> tuple[int, int]:
    temp = a
    a = b
    b = temp
    return a, b
""",
        "wrong_return_variable", 5,
        "Returns (a, a) instead of (a, b) after swap",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_wrv_002.py",
        """\
def min_max(values: list[float]) -> tuple[float, float]:
    lo = values[0]
    hi = values[0]
    for v in values[1:]:
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    return lo, lo
""",
        """\
def min_max(values: list[float]) -> tuple[float, float]:
    lo = values[0]
    hi = values[0]
    for v in values[1:]:
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    return lo, hi
""",
        "wrong_return_variable", 9,
        "Returns (lo, lo) instead of (lo, hi)",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_wrv_003.py",
        """\
def split_evens_odds(nums: list[int]) -> tuple[list[int], list[int]]:
    evens: list[int] = []
    odds: list[int] = []
    for n in nums:
        if n % 2 == 0:
            evens.append(n)
        else:
            odds.append(n)
    return evens, evens
""",
        """\
def split_evens_odds(nums: list[int]) -> tuple[list[int], list[int]]:
    evens: list[int] = []
    odds: list[int] = []
    for n in nums:
        if n % 2 == 0:
            evens.append(n)
        else:
            odds.append(n)
    return evens, odds
""",
        "wrong_return_variable", 9,
        "Returns (evens, evens) instead of (evens, odds)",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_wrv_004.py",
        """\
def compute_stats(data: list[float]) -> tuple[float, float]:
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = variance ** 0.5
    return mean, mean
""",
        """\
def compute_stats(data: list[float]) -> tuple[float, float]:
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = variance ** 0.5
    return mean, std
""",
        "wrong_return_variable", 4,
        "Returns (mean, mean) instead of (mean, std)",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_wrv_005.py",
        """\
def partition(items: list[int], pivot: int) -> tuple[list[int], list[int]]:
    left = [x for x in items if x < pivot]
    right = [x for x in items if x >= pivot]
    return left, left
""",
        """\
def partition(items: list[int], pivot: int) -> tuple[list[int], list[int]]:
    left = [x for x in items if x < pivot]
    right = [x for x in items if x >= pivot]
    return left, right
""",
        "wrong_return_variable", 3,
        "Returns (left, left) instead of (left, right)",
        False, False, False, "held_out",
    ),

    # ── wrong_operator (5 files) ────────────────────────────────────────────────
    _FileSpec(
        "ood_wop_001.py",
        """\
def is_between(value: float, lo: float, hi: float) -> bool:
    return lo <= value and value <= lo
""",
        """\
def is_between(value: float, lo: float, hi: float) -> bool:
    return lo <= value and value <= hi
""",
        "wrong_operator", 2,
        "Upper bound uses 'lo' instead of 'hi'",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_wop_002.py",
        """\
def celsius_to_fahrenheit(c: float) -> float:
    return c * 9 / 5 + 32

def fahrenheit_to_celsius(f: float) -> float:
    return (f + 32) * 5 / 9
""",
        """\
def celsius_to_fahrenheit(c: float) -> float:
    return c * 9 / 5 + 32

def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32) * 5 / 9
""",
        "wrong_operator", 4,
        "fahrenheit_to_celsius uses + instead of - before 32",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_wop_003.py",
        """\
def normalize(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    return [(v - lo) / (hi + lo) for v in values]
""",
        """\
def normalize(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    return [(v - lo) / (hi - lo) for v in values]
""",
        "wrong_operator", 3,
        "Denominator uses (hi + lo) instead of (hi - lo)",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_wop_004.py",
        """\
def compound_interest(principal: float, rate: float, years: int) -> float:
    return principal * (1 - rate) ** years
""",
        """\
def compound_interest(principal: float, rate: float, years: int) -> float:
    return principal * (1 + rate) ** years
""",
        "wrong_operator", 2,
        "Uses (1 - rate) instead of (1 + rate) for growth",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_wop_005.py",
        """\
def manhattan_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return abs(x2 + x1) + abs(y2 - y1)
""",
        """\
def manhattan_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return abs(x2 - x1) + abs(y2 - y1)
""",
        "wrong_operator", 2,
        "x-delta uses + instead of - giving wrong distance",
        False, False, False, "held_out",
    ),

    # ── missing_edge_case (5 files) ─────────────────────────────────────────────
    _FileSpec(
        "ood_mec_001.py",
        """\
def find_median(numbers: list[float]) -> float:
    sorted_nums = sorted(numbers)
    mid = len(sorted_nums) // 2
    return sorted_nums[mid]
""",
        """\
def find_median(numbers: list[float]) -> float:
    if not numbers:
        return 0.0
    sorted_nums = sorted(numbers)
    mid = len(sorted_nums) // 2
    if len(sorted_nums) % 2 == 0:
        return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2.0
    return sorted_nums[mid]
""",
        "missing_edge_case", 1,
        "No empty-list guard; even-length list returns wrong median",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_mec_002.py",
        """\
def safe_head(items: list[object]) -> object:
    return items[0]
""",
        """\
def safe_head(items: list[object]) -> object | None:
    if not items:
        return None
    return items[0]
""",
        "missing_edge_case", 2,
        "No empty-list guard; raises IndexError on empty input",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_mec_003.py",
        """\
def average(values: list[float]) -> float:
    return sum(values) / len(values)
""",
        """\
def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
""",
        "missing_edge_case", 2,
        "No empty-list guard; raises ZeroDivisionError",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_mec_004.py",
        """\
def longest_word(sentence: str) -> str:
    words = sentence.split()
    return max(words, key=len)
""",
        """\
def longest_word(sentence: str) -> str:
    words = sentence.split()
    if not words:
        return ""
    return max(words, key=len)
""",
        "missing_edge_case", 3,
        "No empty-sentence guard; max() raises ValueError on empty sequence",
        False, False, False, "held_out",
    ),
    _FileSpec(
        "ood_mec_005.py",
        """\
def mode(values: list[int]) -> int:
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k])
""",
        """\
def mode(values: list[int]) -> int | None:
    if not values:
        return None
    counts: dict[int, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k])
""",
        "missing_edge_case", 5,
        "No empty-list guard; max() raises ValueError on empty dict",
        False, False, False, "held_out",
    ),
]

_VALIDATION_FILES: list[tuple[str, str]] = [
    (
        "val_001.py",
        """\
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
""",
    ),
    (
        "val_002.py",
        """\
def is_palindrome(text: str) -> bool:
    cleaned = "".join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]
""",
    ),
    (
        "val_003.py",
        """\
def flatten(nested: list[list[int]]) -> list[int]:
    result: list[int] = []
    for sublist in nested:
        result.extend(sublist)
    return result
""",
    ),
    (
        "val_004.py",
        """\
def count_words(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for word in text.lower().split():
        counts[word] = counts.get(word, 0) + 1
    return counts
""",
    ),
    (
        "val_005.py",
        """\
def binary_search(items: list[int], target: int) -> int:
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",
    ),
    (
        "val_006.py",
        """\
def deduplicate(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
""",
    ),
]


def _write_file(path: str, content: str) -> None:
    """Write content to path, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def generate_all() -> None:
    """Generate all data files: training, held-out, validation, ground_truth.json."""
    ground_truth: list[dict] = []

    for spec in _TRAINING_SPECS:
        bug_path = os.path.join(TRAINING_BUGS_DIR, spec.filename)
        clean_path = os.path.join(TRAINING_CLEAN_DIR, spec.filename)
        _write_file(bug_path, spec.bug_code)
        _write_file(clean_path, spec.clean_code)
        ground_truth.append({
            "file_path": bug_path,
            "bug_type": spec.bug_type,
            "bug_line": spec.bug_line,
            "bug_description": spec.bug_description,
            "detectable_by_pylint": spec.detectable_by_pylint,
            "detectable_by_ast": spec.detectable_by_ast,
            "detectable_by_execution": spec.detectable_by_execution,
            "split": spec.split,
        })

    for spec in _HELD_OUT_SPECS:
        bug_path = os.path.join(HELD_OUT_BUGS_DIR, spec.filename)
        clean_path = os.path.join(HELD_OUT_CLEAN_DIR, spec.filename)
        _write_file(bug_path, spec.bug_code)
        _write_file(clean_path, spec.clean_code)
        ground_truth.append({
            "file_path": bug_path,
            "bug_type": spec.bug_type,
            "bug_line": spec.bug_line,
            "bug_description": spec.bug_description,
            "detectable_by_pylint": spec.detectable_by_pylint,
            "detectable_by_ast": spec.detectable_by_ast,
            "detectable_by_execution": spec.detectable_by_execution,
            "split": spec.split,
        })

    for filename, content in _VALIDATION_FILES:
        val_path = os.path.join(VALIDATION_DIR, filename)
        _write_file(val_path, content)

    os.makedirs(os.path.dirname(GROUND_TRUTH_PATH), exist_ok=True)
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as fh:
        json.dump(ground_truth, fh, indent=2)

    logger.info(
        "dataset_builder.generate_all.done",
        training=len(_TRAINING_SPECS),
        held_out=len(_HELD_OUT_SPECS),
        validation=len(_VALIDATION_FILES),
        ground_truth_entries=len(ground_truth),
    )
