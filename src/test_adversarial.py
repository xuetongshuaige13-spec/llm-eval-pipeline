import sys
from pathlib import Path

# 强制将项目根目录注入 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import json
from src.eval_pipeline import LLMEvalPipeline, JudgeResult

def run_adversarial_test():
    # 模拟同一问题的三种典型候选回答
    query = "香港科技大学大数据科技硕士申请要求是什么？"
    ground_truth = "通常需要计算机、数学或工程背景，雅思总分不低于6.5分。"

    test_cases = [
        {
            "case_type": "高质量正向回答 (High Quality)",
            "candidate": "申请该专业一般要求申请人具备计算机、数学、统计或工程等相关量化背景，语言成绩方面要求雅思总分至少达到6.5分。"
        },
        {
            "case_type": "严重事实幻觉 (Low Faithfulness)",
            "candidate": "该专业没有任何专业背景限制，文科艺术生均可随意申请，并且完全不需要雅思或托福成绩，仅凭高中毕业证即可免试录取。"
        },
        {
            "case_type": "答非所问/低关联度 (Low Relevance)",
            "candidate": "香港科技大学依山傍海，位于清水湾，校园风景优美，是香港顶尖的研究型大学之一，食堂饭菜也很不错。"
        }
    ]

    # 初始化评测管线
    pipeline = LLMEvalPipeline(
        input_file=os.path.join("data", "processed", "clean_qa.jsonl"),
        output_file=os.path.join("data", "processed", "eval_adversarial_results.jsonl")
    )

    print("=" * 60)
    print("      LLM 裁判敏感度与区分度压测 (Adversarial Stress Test)")
    print("=" * 60)

    for case in test_cases:
        case_type = case["case_type"]
        candidate = case["candidate"]
        print(f"\n[测试用例类型]: {case_type}")
        print(f"-> 待测候选回答: {candidate}")

        # 调用核心评测函数
        result: JudgeResult = pipeline._call_judge_llm(
            query=query,
            ground_truth=ground_truth,
            predicted_resp=candidate
        )

        print(f"   [评分结果] 关联度: {result.relevance_score}/5 | 忠实度: {result.faithfulness_score}/5 | 是否通过: {result.overall_pass}")
        print(f"   [关联度理由]: {result.relevance_rationale}")
        print(f"   [忠实度理由]: {result.faithfulness_rationale}")

if __name__ == "__main__":
    run_adversarial_test()
