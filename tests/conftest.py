import pytest

from src.verifier.models import GroundTruth


@pytest.fixture
def clean_python_code() -> str:
    """Valid 20-line Python function with no bugs."""
    return """\
def compute_statistics(numbers: list[float]) -> dict[str, float]:
    if not numbers:
        return {"mean": 0.0, "variance": 0.0, "std_dev": 0.0}
    n = len(numbers)
    mean = sum(numbers) / n
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std_dev = variance ** 0.5
    return {"mean": mean, "variance": variance, "std_dev": std_dev}


def find_median(numbers: list[float]) -> float:
    if not numbers:
        return 0.0
    sorted_nums = sorted(numbers)
    mid = len(sorted_nums) // 2
    if len(sorted_nums) % 2 == 0:
        return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2.0
    return sorted_nums[mid]
"""


@pytest.fixture
def undefined_variable_code() -> str:
    """Code with E0602 undefined variable bug — function is called to trigger runtime NameError."""
    return """\
def process_data(items: list[int]) -> list[int]:
    result = []
    for item in items:
        result.append(item * multiplier)
    return result

process_data([1, 2, 3])
"""


@pytest.fixture
def mutable_default_code() -> str:
    """Code with mutable default argument anti-pattern."""
    return """\
def append_item(value: int, collection: list[int] = []) -> list[int]:
    collection.append(value)
    return collection
"""


@pytest.fixture
def bare_except_code() -> str:
    """Code with bare except clause."""
    return """\
def safe_divide(a: float, b: float) -> float:
    try:
        return a / b
    except:
        return 0.0
"""


@pytest.fixture
def off_by_one_code() -> str:
    """Code with subtle off-by-one logic error pylint cannot catch."""
    return """\
def find_last_index(items: list[int], target: int) -> int:
    for i in range(len(items) - 1):
        if items[i] == target:
            return i
    return -1
"""


@pytest.fixture
def wrong_return_variable_code() -> str:
    """Code with wrong return variable — logic error pylint cannot catch."""
    return """\
def swap_values(a: int, b: int) -> tuple[int, int]:
    temp = a
    a = b
    b = temp
    return a, a
"""


@pytest.fixture
def ground_truth_undefined() -> GroundTruth:
    """Ground truth for undefined_variable_code."""
    return GroundTruth(
        file_path="<test>",
        bug_type="undefined_variable",
        bug_line=4,
        bug_description="Variable 'multiplier' used before assignment",
        detectable_by_pylint=True,
        detectable_by_ast=False,
        detectable_by_execution=True,
    )


@pytest.fixture
def ground_truth_logic() -> GroundTruth:
    """Ground truth for off_by_one_code — not detectable by any verifier."""
    return GroundTruth(
        file_path="<test>",
        bug_type="off_by_one",
        bug_line=2,
        bug_description="range(len(items) - 1) skips the last element",
        detectable_by_pylint=False,
        detectable_by_ast=False,
        detectable_by_execution=False,
    )


@pytest.fixture
def ground_truth_mutable_default() -> GroundTruth:
    """Ground truth for mutable_default_code."""
    return GroundTruth(
        file_path="<test>",
        bug_type="mutable_default_argument",
        bug_line=1,
        bug_description="Mutable default argument [] will persist across calls",
        detectable_by_pylint=True,
        detectable_by_ast=True,
        detectable_by_execution=False,
    )


@pytest.fixture
def ground_truth_bare_except() -> GroundTruth:
    """Ground truth for bare_except_code."""
    return GroundTruth(
        file_path="<test>",
        bug_type="bare_except",
        bug_line=4,
        bug_description="Bare except catches all exceptions including KeyboardInterrupt",
        detectable_by_pylint=True,
        detectable_by_ast=True,
        detectable_by_execution=False,
    )
