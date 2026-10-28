import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ================================
# 1. RAG 评测输出结构 Schema
# ================================
class RAGTriadMetric(BaseModel):
    """RAG 三元组评测核心结果"""
    context_relevance_score: int = Field(..., ge=1, le=5, description="检索上下文与问题的相关度 (1-5)")
    context_relevance_rationale: str = Field(..., description="上下文相关度分析理由")

    faithfulness_score: int = Field(..., ge=1, le=5, description="回答基于检索事实的忠实度 (1-5)")
    faithfulness_rationale: str = Field(..., description="忠实度与幻觉分析理由")
    hallucinated_claims: List[str] = Field(default=[], description="识别出的未被上下文支撑的幻觉断言句子")

    answer_relevance_score: int = Field(..., ge=1, le=5, description="回答与问题的关联度 (1-5)")
    answer_relevance_rationale: str = Field(..., description="回答相关度分析理由")

    overall_pass: bool = Field(..., description="三项打分是否均达到 3 分及以上")


class RAGEvalRecord(BaseModel):
    id: str
    query: str
    contexts: List[str]
    response: str
    eval_result: RAGTriadMetric


# ================================
# 2. RAG 评测 Prompt 设计
# ================================
RAG_JUDGE_PROMPT = """你是一个严谨的企业级 RAG（检索增强生成）系统效果评测专家。
请根据给定的 [用户问题]、[检索召回切片 contexts] 和 [模型生成的回答 response]，从以下三个正交维度进行打分：

1. Context Relevance (检索相关度, 1-5分):
   - 检索到的上下文片段是否包含回答该问题所必需的事实信息？如果片段通篇是无关噪音，给 1-2 分；如果高度契合，给 4-5 分。
2. Faithfulness / Groundedness (忠实度与幻觉, 1-5分):
   - 回答中的所有陈述，是否能够**完全且无歧义地从检索上下文中推导出来**？
   - 严禁大模型引入检索片段以外的“先验常识”作为事实；若回答包含上下文未提及或与上下文冲突的断言，判为幻觉并大幅扣分（1-2分），并在 hallucinated_claims 中列出具体断言。
3. Answer Relevance (回答相关度, 1-5分):
   - 回答是否切中用户的问题，是否存在答非所问、过度冗长或避而不答。

输出必须严格为合法 JSON 格式，字段定义如下：
{
  "context_relevance_score": <1-5整数>,
  "context_relevance_rationale": "<详细理由>",
  "faithfulness_score": <1-5整数>,
  "faithfulness_rationale": "<详细理由>",
  "hallucinated_claims": ["<幻觉句子1>", "<幻觉句子2>"],
  "answer_relevance_score": <1-5整数>,
  "answer_relevance_rationale": "<详细理由>",
  "overall_pass": <true/false, 三项均>=3分时为true>
}
"""

# ================================
# 3. 评测引擎实现
# ================================
class RAGTriadPipeline:
    def __init__(self, input_file: str, output_file: str, max_concurrency: int = 3):
        self.input_file = input_file
        self.output_file = output_file
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if not self.mock_mode:
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            )
            self.model_name = os.getenv("MODEL_NAME", "deepseek-chat")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _evaluate_rag(self, query: str, contexts: List[str], response: str) -> RAGTriadMetric:
        user_input = f"""[用户问题]: {query}
[检索召回切片 contexts]:
{json.dumps(contexts, ensure_ascii=False, indent=2)}

[模型生成的回答 response]:
{response}
"""
        res = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": RAG_JUDGE_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = res.choices[0].message.content
        if not content:
            raise ValueError("裁判大模型返回为空")
        return RAGTriadMetric.model_validate_json(content)

    async def _process_item(self, record: Dict[str, Any]) -> RAGEvalRecord | None:
        async with self.semaphore:
            item_id = str(record.get("id"))
            query = record["query"]
            contexts = record.get("contexts", [])
            resp = record["response"]

            print(f"[*] [正在评测 RAG 样本] [{item_id}]: {query[:25]}...")
            try:
                metric = await self._evaluate_rag(query, contexts, resp)
                return RAGEvalRecord(
                    id=item_id,
                    query=query,
                    contexts=contexts,
                    response=resp,
                    eval_result=metric
                )
            except Exception as e:
                print(f"[!] 样本 [{item_id}] 评测异常: {str(e)}")
                return None

    async def run(self):
        items: List[Dict[str, Any]] = []
        with open(self.input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line.strip()))

        tasks = [self._process_item(item) for item in items]
        records = [r for r in await asyncio.gather(*tasks) if r is not None]

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as fout:
            for r in records:
                fout.write(r.model_dump_json() + "\n")

        self._print_dashboard(records)

    def _print_dashboard(self, records: List[RAGEvalRecord]):
        total = len(records)
        if total == 0:
            return

        avg_ctx = sum(r.eval_result.context_relevance_score for r in records) / total
        avg_faith = sum(r.eval_result.faithfulness_score for r in records) / total
        avg_ans = sum(r.eval_result.answer_relevance_score for r in records) / total
        pass_rate = (sum(1 for r in records if r.eval_result.overall_pass) / total) * 100

        print("\n" + "=" * 55)
        print("          RAG 三元组与幻觉评测看板 (RAG Triad Dashboard)")
        print("=" * 55)
        print(f"评测样本总量 (Evaluated Samples):        {total}")
        print(f"平均检索相关度 (Avg Context Relevance):   {avg_ctx:.2f} / 5.0")
        print(f"平均回答忠实度 (Avg Faithfulness):        {avg_faith:.2f} / 5.0")
        print(f"平均回答相关度 (Avg Answer Relevance):     {avg_ans:.2f} / 5.0")
        print(f"RAG 系统综合合格率 (Pass Rate):           {pass_rate:.2f}%")
        print("=" * 55)
        print(f"[+] 评测结果明细已写入: {self.output_file}\n")


if __name__ == "__main__":
    INPUT = os.path.join("data", "raw", "sample_rag_eval.jsonl")
    OUTPUT = os.path.join("data", "processed", "rag_eval_results.jsonl")

    pipeline = RAGTriadPipeline(input_file=INPUT, output_file=OUTPUT, max_concurrency=3)
    asyncio.run(pipeline.run())
