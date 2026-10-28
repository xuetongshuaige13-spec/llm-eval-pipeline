import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.eval_pipeline import AgentEvalSample, ToolCallSpec, AgentToolEvaluator

def test_tool_call_exact_match():
    sample = AgentEvalSample(
        id="test_1",
        user_query="查天气",
        expected_call=ToolCallSpec(name="weather", arguments={"city": "深圳"}),
        model_output_call=ToolCallSpec(name="weather", arguments={"city": "深圳"})
    )
    result = AgentToolEvaluator.evaluate_sample(sample)
    assert result.passed is True
    assert result.tool_selection_correct is True
    assert result.argument_precision == 1.0

def test_tool_over_triggering_detected():
    sample = AgentEvalSample(
        id="test_2",
        user_query="你好",
        expected_call=None,
        model_output_call=ToolCallSpec(name="search", arguments={})
    )
    result = AgentToolEvaluator.evaluate_sample(sample)
    assert result.passed is False
    assert "误触发" in result.rationale
