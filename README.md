# 基于 Python 的生成式 AI 问答数据清洗与自动化评测管线 (LLM Data Clean & Eval Pipeline)
[![CI Pipeline](https://github.com/xuetongshuaige13-spec/llm-eval-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/xuetongshuaige13-spec/llm-eval-pipeline/actions/workflows/ci.yml)
面向企业级大模型问答系统的轻量化工程管线。涵盖从原始弱格式/脏数据的声明式模式清洗，到基于 **LLM-as-a-Judge** 的多维度自动化评估，支持 **RAG 三元组** 评估、原子级幻觉定位与高并发异步执行，并提供交互式 Web 审计看板。

<p align="center">
  <img src="assets/dashboard_preview.png" alt="RAG Eval Dashboard Preview" width="100%">
</p>

---

## 核心架构设计

```text
[原始脏数据 (JSONL)]
         │
         ▼
[DataCleanPipeline] ─── (流式解析 + Pydantic V2 强模式约束 + 业务清洗)
         │
         ├──> [清洗审计报告 (Audit Log)]
         └──> [标准 QA 数据集 (clean_qa.jsonl)]
                     │
                     ▼
         [AsyncLLMEvalPipeline / RAGTriadPipeline] ─── (AsyncOpenAI + 并发信号量 + tenacity 重试)
                     │
                     ├──> 维度 1: 检索相关度 (Context Relevance, 1-5)
                     ├──> 维度 2: 回答忠实度 (Faithfulness, 1-5) ──> [原子级幻觉定位]
                     └──> 维度 3: 回答相关度 (Answer Relevance, 1-5)
                     │
                     ▼
         [Streamlit 交互审计看板 (app.py) & 评测结果明细 (JSONL)]
```

---

## 核心技术特性

1. **声明式数据清洗 (Declarative Sanitization)**：
   - 基于 **Pydantic V2** 构建标准数据模型约束。
   - 提供流式容错解析，自动隔离 `JSONDecodeError`、非法字段类型与空白异常数据。
   - 端到端审计追踪，量化计算清洗合格率（Pass Rate）。
2. **高并发异步评测引擎 (Async Evaluation Engine)**：
   - 基于 `asyncio` 与 `AsyncOpenAI` 搭建非阻塞评测执行流水线。
   - 使用 `asyncio.Semaphore` 实施精细化并发控制，杜绝瞬时流量触发 API 限流（429）。
   - 接入 `tenacity` 库实现带随机抖动的指数退避重试（Exponential Backoff with Jitter）。
3. **RAG 三元组与细粒度幻觉排查 (RAG Triad & Hallucination)**：
   - **Context Relevance**：评估检索召回切片的支撑有效性与噪声比例。
   - **Faithfulness / Groundedness**：基于原子断言（Atomic Claims）拆解，检测大模型先验知识污染，并定位幻觉句子。
   - **Answer Relevance**：评估模型回答与用户真实意图的契合程度。
4. **Streamlit 交互式审计看板**：
   - 提供开箱即用的 Web 评测工作台，集成综合合格率、维度均值分布及 KPI 指标卡。
   - 支持样本下钻筛选，高亮展示被系统捕获的幻觉断言及裁判推理（Rationale）。

---

## 项目目录结构

```text
llm-eval-pipeline/
├── assets/
│   └── dashboard_preview.png # 看板效果预览截图
├── data/
│   ├── raw/                 # 原始脏数据与 RAG 评测输入
│   └── processed/           # 清洗入库及评测打分结果
├── src/
│   ├── __init__.py          # 模块标识
│   ├── app.py               # Streamlit 可视化审计看板
│   ├── clean_pipeline.py    # 数据清洗与校验流水线
│   ├── eval_pipeline.py     # 异步并发评测流水线
│   ├── rag_eval_pipeline.py # RAG 三元组与幻觉评测流水线
│   └── test_adversarial.py  # 裁判区分度对抗压测脚本
├── .env.example             # 环境变量配置模板
├── .gitignore               # Git 忽略规则
└── requirements.txt         # 核心依赖清单
```

---

## 快速上手

### 1. 虚拟环境配置
```bash
git clone https://github.com/xuetongshuaige13-spec/llm-eval-pipeline.git
cd llm-eval-pipeline
python -m venv venv

# Windows PowerShell 激活
.\venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量
复制环境模板并填入兼容 OpenAI 接口的大模型 API 凭证：
```bash
copy .env.example .env
```
修改 `.env` 核心配置：
```ini
OPENAI_API_KEY=your_actual_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
MOCK_MODE=false
```

### 3. 执行核心流水线
```bash
# 运行数据清洗管线
python src/clean_pipeline.py

# 运行异步自动化评测
python src/eval_pipeline.py

# 运行 RAG 三元组与幻觉评测
python src/rag_eval_pipeline.py

# 运行对抗样本区分度测试
python src/test_adversarial.py
```

### 4. 启动 Web 可视化交互看板
```bash
streamlit run src/app.py
```

---

## RAG 评测压测实测指标

| 样本类型 | 检索相关度 (Context) | 回答忠实度 (Faithfulness) | 回答相关度 (Answer) | 幻觉检测定位 | 综合判定 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 黄金基准样本 (`rag_001`) | 5.0 / 5 | 5.0 / 5 | 5.0 / 5 | 无幻觉 | **Pass** |
| 检索噪声/上下文缺失 (`rag_002`) | 2.0 / 5 | 2.0 / 5 | 4.0 / 5 | 上下文未支撑断言 | **Failed** |
| 事实严重冲突/强幻觉 (`rag_003`) | 4.0 / 5 | 1.0 / 5 | 3.0 / 5 | 精准提取虚构断言 | **Failed** |
