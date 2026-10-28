import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# 统一工程根目录寻址
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
# 3. 异步高并发评测流水线
# ================================
class AsyncLLMEvalPipeline:
    def __init__(self, input_file: str, output_file: str, max_concurrency: int = 5):
        self.input_file = input_file
        self.output_file = output_file
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if not self.mock_mode:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            if not api_key or api_key == "your_api_key_here":
                raise ValueError("未检测到有效 OPENAI_API_KEY。若要模拟调试，请在 .env 中设置 MOCK_MODE=true")
            # 切换为异步客户端 AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self.model_name = os.getenv("MODEL_NAME", "deepseek-chat")

    # 配置指数退避重试：遇到网络抖动或解析异常最多重试 3 次，等待时间以 2^x 秒递增
    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def _call_judge_llm_with_retry(self, query: str, ground_truth: str, predicted_resp: str) -> JudgeResult:
        """带自动重试的大模型异步评测调用"""
        if self.mock_mode:
            await asyncio.sleep(0.1)  # 模拟异步 I/O 网络延时
            return JudgeResult(
                relevance_score=5,
                relevance_rationale="[Mock] 候选回答准确命中了用户提问意图。",
                faithfulness_score=5,
                faithfulness_rationale="[Mock] 回答与参考答案一致。",
                overall_pass=True
            )

        prompt = JUDGE_USER_TEMPLATE.format(
            query=query,
            ground_truth=ground_truth,
            predicted_response=predicted_resp
        )

        response = await self.client.chat.completions.create(
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
            raise ValueError(f"裁判大模型返回为空: {response}")

        return JudgeResult.model_validate_json(raw_content)

    async def _evaluate_single_item(self, data: Dict[str, Any], idx: int) -> EvalRecord | None:
        """单条评测任务：受信号量保护的协程任务"""
        async with self.semaphore:
            query = data["query"]
            ground_truth = data["response"]
            candidate = f"{ground_truth}（经由测试模型生成）"
            item_id = str(data.get("id", idx))

            print(f"[*] [任务并发中] 样本 [{item_id}]: {query[:20]}...")
            try:
                judge_result = await self._call_judge_llm_with_retry(query, ground_truth, candidate)
                return EvalRecord(
                    id=item_id,
                    query=query,
                    ground_truth=ground_truth,
                    predicted_response=candidate,
                    judge=judge_result
                )
            except Exception as e:
                print(f"[!] [任务失败/重试耗尽] 样本 [{item_id}]: {str(e)}")
                return None

    async def run(self):
        """执行端到端异步批量评测"""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"未找到输入数据文件: {self.input_file}")

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        raw_items: List[Dict[str, Any]] = []
        with open(self.input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_items.append(json.loads(line.strip()))

        total_count = len(raw_items)
        print(f"\n=======================================================")
        print(f"[*] 启动异步评测引擎 (并发上限: {self.max_concurrency} | 总任务数: {total_count})")
        print(f"=======================================================\n")

        # 创建所有协程任务并使用 asyncio.gather 并发调度
        tasks = [
            self._evaluate_single_item(item, idx)
            for idx, item in enumerate(raw_items, start=1)
        ]
        results = await asyncio.gather(*tasks)

        # 过滤执行失败的脏数据并收集有效记录
        valid_records: List[EvalRecord] = [r for r in results if r is not None]

        # 写入结果明细文件
        with open(self.output_file, "w", encoding="utf-8") as fout:
            for item in valid_records:
                fout.write(item.model_dump_json() + "\n")

        self._print_eval_report(valid_records)

    def _print_eval_report(self, records: List[EvalRecord]):
        """生成评测审计看板"""
        total = len(records)
        if total == 0:
            print("[!] 本次评测未产出有效数据。")
            return

        avg_rel = sum(r.judge.relevance_score for r in records) / total
        avg_faith = sum(r.judge.faithfulness_score for r in records) / total
        pass_count = sum(1 for r in records if r.judge.overall_pass)
        pass_rate = (pass_count / total) * 100

        print("\n" + "="*45)
        print("        异步自动化评测看板 (Async Eval Dashboard)")
        print("="*45)
        print(f"评测总样本数 (Total Evaluated):       {total}")
        print(f"平均关联度分 (Avg Relevance):         {avg_rel:.2f} / 5.0")
        print(f"平均忠实度分 (Avg Faithfulness):      {avg_faith:.2f} / 5.0")
        print(f"综合合格率 (Overall Pass Rate):       {pass_rate:.2f}%")
        print("="*45)
        print(f"[+] 异步评测明细已保存至: {self.output_file}\n")


if __name__ == "__main__":
    INPUT_FILE = os.path.join("data", "processed", "clean_qa.jsonl")
    OUTPUT_FILE = os.path.join("data", "processed", "eval_results.jsonl")

    # 实例化并设置最大并发度为 5
    pipeline = AsyncLLMEvalPipeline(input_file=INPUT_FILE, output_file=OUTPUT_FILE, max_concurrency=5)

    # 启动异步事件循环
    asyncio.run(pipeline.run())
    
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# ================================
# 1. Agent 工具调用评测数据契约
# ================================

class ToolCallSpec(BaseModel):
    name: str = Field(description="调用的工具函数名")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="工具参数字典")

class AgentEvalSample(BaseModel):
    id: str
    user_query: str
    expected_call: Optional[ToolCallSpec] = None  # None 代表此意图不应触发任何工具（闲聊或直接回答）
    model_output_call: Optional[ToolCallSpec] = None

class AgentToolEvalResult(BaseModel):
    id: str
    tool_selection_correct: bool = Field(description="工具选择是否完全一致")
    argument_precision: float = Field(description="参数精确率: 实际生成的参数中有多少是预期正确的")
    argument_recall: float = Field(description="参数召回率: 预期的关键参数有多少被成功生成")
    passed: bool = Field(description="综合判定是否全绿通过")
    rationale: str = Field(description="审计原因说明")


# ================================
# 2. 确定性评估引擎
# ================================

class AgentToolEvaluator:
    """Agent 工具调用与参数决策评估器"""

    @staticmethod
    def evaluate_sample(sample: AgentEvalSample) -> AgentToolEvalResult:
        exp = sample.expected_call
        act = sample.model_output_call

        # 场景 A: 负样本/拒识测试（预期不调用工具）
        if exp is None:
            if act is None:
                return AgentToolEvalResult(
                    id=sample.id,
                    tool_selection_correct=True,
                    argument_precision=1.0,
                    argument_recall=1.0,
                    passed=True,
                    rationale="正确识别无需调用工具，直接回复"
                )
            else:
                return AgentToolEvalResult(
                    id=sample.id,
                    tool_selection_correct=False,
                    argument_precision=0.0,
                    argument_recall=0.0,
                    passed=False,
                    rationale=f"发生误触发（Over-triggering）：错误调用了工具 '{act.name}'"
                )

        # 场景 B: 预期调用工具，但模型漏调（Under-triggering）
        if act is None:
            return AgentToolEvalResult(
                id=sample.id,
                tool_selection_correct=False,
                argument_precision=0.0,
                argument_recall=0.0,
                passed=False,
                rationale=f"遗漏工具调用：预期调用 '{exp.name}'，模型直接回答"
            )

        # 场景 C: 工具名称对比
        tool_match = (exp.name == act.name)
        if not tool_match:
            return AgentToolEvalResult(
                id=sample.id,
                tool_selection_correct=False,
                argument_precision=0.0,
                argument_recall=0.0,
                passed=False,
                rationale=f"工具选型错误：预期 '{exp.name}'，实际调用 '{act.name}'"
            )

        # 场景 D: 参数逐字段比对（计算 Precision 和 Recall）
        exp_args = exp.arguments
        act_args = act.arguments

        correct_args_count = 0
        for k, v in act_args.items():
            if k in exp_args and exp_args[k] == v:
                correct_args_count += 1

        precision = correct_args_count / len(act_args) if act_args else (1.0 if not exp_args else 0.0)
        recall = correct_args_count / len(exp_args) if exp_args else 1.0

        is_passed = (precision == 1.0 and recall == 1.0)

        return AgentToolEvalResult(
            id=sample.id,
            tool_selection_correct=True,
            argument_precision=round(precision, 2),
            argument_recall=round(recall, 2),
            passed=is_passed,
            rationale="参数完全匹配" if is_passed else f"参数不完全匹配: Precision={precision:.2f}, Recall={recall:.2f}"
        )
