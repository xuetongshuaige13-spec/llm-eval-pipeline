import json
import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# 加载 .env 环境变量
load_dotenv()

# ================================
# 1. 评测输出 Schema 定义
# ================================
class JudgeResult(BaseModel):
    """裁判模型的单条评测结构化输出"""
    relevance_score: int = Field(..., ge=1, le=5, description="回答与问题的关联度得分 (1-5)")
    relevance_rationale: str = Field(..., description="关联度打分理由")
    faithfulness_score: int = Field(..., ge=1, le=5, description="回答与参考事实的忠实度得分 (1-5)")
    faithfulness_rationale: str = Field(..., description="忠实度打分理由")
    overall_pass: bool = Field(..., description="综合评分是否通过 (两项得分均 >= 3 则为 True)")


class EvalRecord(BaseModel):
    """落盘保存的完整评测记录"""
    id: str
    query: str
    ground_truth: str
    predicted_response: str
    judge: JudgeResult


# ================================
# 2. 评测 Prompt 模版
# ================================
JUDGE_SYSTEM_PROMPT = """你是一个严谨的大语言模型问答评测专家。
你需要根据给定的 [用户问题] 和 [标准参考答案]，对待评测的 [候选回答] 进行打分。

打分维度如下：
1. Relevance (关联度, 1-5分): 候选回答是否正面且清晰地回应了用户问题。
   - 1分: 完全无关或答非所问
   - 3分: 基本回答了问题，但包含较多冗余或不够聚焦
   - 5分: 完美回应问题核心，逻辑清晰
2. Faithfulness (忠实度, 1-5分): 候选回答是否遵循标准参考答案中的事实，是否存在虚构或事实冲突（幻觉）。
   - 1分: 存在严重事实错误或严重幻觉
   - 3分: 核心事实大致正确，但有次要事实缺失或轻微模糊
   - 5分: 完全符合事实，无任何虚构

你必须严格以合法的 JSON 格式返回结果，禁止附加任何 Markdown 标签外的解释，字段格式如下：
{
  "relevance_score": <1-5整数>,
  "relevance_rationale": "<关联度打分理由>",
  "faithfulness_score": <1-5整数>,
  "faithfulness_rationale": "<忠实度打分理由>",
  "overall_pass": <true/false, 两项得分均>=3为true>
}
"""

JUDGE_USER_TEMPLATE = """[用户问题]: {query}
[标准参考答案]: {ground_truth}
[待评测候选回答]: {predicted_response}
"""


# ================================
# 3. 评测流水线核心类
# ================================
class LLMEvalPipeline:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if not self.mock_mode:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            if not api_key or api_key == "your_api_key_here":
                raise ValueError("未检测到有效 OPENAI_API_KEY。若要进行本地模拟调试，请在 .env 中设置 MOCK_MODE=true")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

    def _call_judge_llm(self, query: str, ground_truth: str, predicted_resp: str) -> JudgeResult:
        """调用大模型裁判或本地 Mock 评测"""
        if self.mock_mode:
            # 本地 Mock 数据，保证无 Key 状态下全管线通畅跑通
            return JudgeResult(
                relevance_score=5,
                relevance_rationale="[Mock] 候选回答准确命中了用户提问的核心实体与意图。",
                faithfulness_score=4,
                faithfulness_rationale="[Mock] 回答与标准答案基本吻合，未检测到明显事实幻觉。",
                overall_pass=True
            )

        prompt = JUDGE_USER_TEMPLATE.format(
            query=query,
            ground_truth=ground_truth,
            predicted_response=predicted_resp
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        raw_content = response.choices[0].message.content
        if not raw_content:
            raise ValueError(f"裁判大模型返回内容为空或被拦截: {response}")

        return JudgeResult.model_validate_json(raw_content)

    def run(self):
        """执行批量评测"""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"未找到输入数据文件: {self.input_file}")

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        eval_records: List[EvalRecord] = []

        print(f"[*] 启动评测流程 (Mock Mode: {self.mock_mode})")
        print(f"[*] 读取清洗后数据: {self.input_file}")

        with open(self.input_file, "r", encoding="utf-8") as fin:
            for idx, line in enumerate(fin, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue

                data = json.loads(clean_line)
                query = data["query"]
                ground_truth = data["response"]

                # 构造候选回答（实际项目中来自待测试模型输出；此处模拟生成略有变动的测试回答）
                candidate_response = f"{ground_truth}（本回答由测试模型生成）"

                print(f"    -> 正在评测样本 [{data.get('id', idx)}]: {query[:25]}...")
                try:
                    judge_result = self._call_judge_llm(query, ground_truth, candidate_response)
                    record = EvalRecord(
                        id=data.get("id", str(idx)),
                        query=query,
                        ground_truth=ground_truth,
                        predicted_response=candidate_response,
                        judge=judge_result
                    )
                    eval_records.append(record)
                except (ValidationError, Exception) as e:
                    print(f"    [!] 评测出错: {str(e)}")

        # 保存评测结果
        with open(self.output_file, "w", encoding="utf-8") as fout:
            for item in eval_records:
                fout.write(item.model_dump_json() + "\n")

        self._print_eval_report(eval_records)

    def _print_eval_report(self, records: List[EvalRecord]):
        """生成并输出评测统计指标看板"""
        total = len(records)
        if total == 0:
            print("[!] 未产出有效评测数据。")
            return

        avg_rel = sum(r.judge.relevance_score for r in records) / total
        avg_faith = sum(r.judge.faithfulness_score for r in records) / total
        pass_count = sum(1 for r in records if r.judge.overall_pass)
        pass_rate = (pass_count / total) * 100

        print("\n" + "="*45)
        print("           自动化评测审计看板 (Eval Dashboard)")
        print("="*45)
        print(f"评测样本总数 (Total Evaluated):       {total}")
        print(f"平均关联度分 (Avg Relevance):         {avg_rel:.2f} / 5.0")
        print(f"平均忠实度分 (Avg Faithfulness):      {avg_faith:.2f} / 5.0")
        print(f"综合合格率 (Overall Pass Rate):       {pass_rate:.2f}%")
        print("="*45)
        print(f"[+] 评测明细文件已保存至: {self.output_file}\n")


if __name__ == "__main__":
    INPUT_FILE = os.path.join("data", "processed", "clean_qa.jsonl")
    OUTPUT_FILE = os.path.join("data", "processed", "eval_results.jsonl")

    pipeline = LLMEvalPipeline(input_file=INPUT_FILE, output_file=OUTPUT_FILE)
    pipeline.run()
