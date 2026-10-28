import os
import sys
from pathlib import Path
import pytest
from pydantic import ValidationError

# 注入项目根目录路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# 从清洗流水线导入真实类名
from src.clean_pipeline import StandardQAPair, DataCleanPipeline


# ================================
# 1. 模式校验与边界拦截用例
# ================================

def test_valid_qa_pair_parsing():
    """测试合法数据是否能正常通过 Pydantic 校验"""
    valid_data = {
        "id": "qa_valid_001",
        "query": "香港科技大学大数据科技硕士项目的学费是多少？",
        "response": "全日制学费为 240,000 港币。"
    }
    item = StandardQAPair(**valid_data)
    assert item.id == "qa_valid_001"
    assert "香港科技大学" in item.query
    assert "240,000" in item.response


def test_missing_required_fields():
    """测试缺失核心字段时必须抛出 ValidationError"""
    corrupted_data = {
        "id": "qa_corrupted_001",
        "query": "只有问题，没有回答"
    }
    with pytest.raises(ValidationError):
        StandardQAPair(**corrupted_data)


# ================================
# 2. 端到端清洗流水线容错用例
# ================================

def test_pipeline_audit_metrics(tmp_path):
    """测试清洗流水线遇到损坏数据时的隔离能力"""
    raw_file = tmp_path / "test_dirty.jsonl"
    clean_file = tmp_path / "test_clean.jsonl"

    test_lines = [
        '{"id": "1", "query": "合法问题1？", "response": "合法回答1"}',
        '{corrupted_json_line}',
        '{"id": "2", "query": "合法问题2？", "response": "合法回答2"}'
    ]
    raw_file.write_text("\n".join(test_lines), encoding="utf-8")

    pipeline = DataCleanPipeline(raw_path=str(raw_file), output_path=str(clean_file))
    pipeline.run()

    assert clean_file.exists()
    cleaned_records = clean_file.read_text(encoding="utf-8").strip().split("\n")
    # 3 条数据中应仅有 2 条合法记录被写入
    assert len(cleaned_records) == 2
