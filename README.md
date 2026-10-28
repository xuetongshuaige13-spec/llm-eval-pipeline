# 基于 Python 的生成式 AI 问答数据清洗与自动化评测管线 (LLM Data Clean & Eval Pipeline)

面向企业级大模型问答系统的轻量化工程管线。涵盖从原始弱格式/脏数据的声明式模式清洗，到基于 **LLM-as-a-Judge** 的多维度自动化评估，并集成了对抗样本区分度测试机制。

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
         [LLMEvalPipeline] ─── (OpenAI 兼容协议 + JSON Mode 强结构输出)
                     │
                     ├──> 维度 1: 关联度 (Relevance, 1-5)
                     ├──> 维度 2: 忠实度 (Faithfulness, 1-5)
                     └──> 幻觉检测 (Hallucination Detection)
                     │
                     ▼
         [评测审计看板 (Eval Dashboard) & 结果明细 (eval_results.jsonl)]
核心技术特性声明式数据清洗 (Declarative Data Sanitization)：基于 Pydantic V2 构建标准 QA Schema 约束。提供流式容错处理机制，优雅隔离 JSONDecodeError、非法字段类型与空白字符串。包含端到端清洗审计追踪，量化计算数据清洗合格率（Pass Rate）。LLM-as-a-Judge 自动化评测打分：将评测准则（Rubric）工程化，解耦为 关联度 (Relevance) 与 忠实度 (Faithfulness) 两个正交维度。利用 LLM 的 response_format={"type": "json_object"} 强制输出结构化评分与打分推理（Rationale）。具备本地 Mock 降级机制，支持无 Key 状态下的全链路逻辑调试。裁判鉴别力对抗压测 (Adversarial Stress Test)：内置高质量回答、严重事实幻觉、答非所问三类对照实验样本。验证评测体系对虚构信息与语义漂移的敏感度，规避模型“老好人”打分偏好。项目目录结构Plaintextllm-eval-pipeline/
├── data/
│   ├── raw/                 # 原始脏数据
│   └── processed/           # 清洗入库及评测打分结果
├── src/
│   ├── __init__.py          # 模块标识
│   ├── clean_pipeline.py    # 数据清洗与校验流水线
│   ├── eval_pipeline.py     # LLM 自动化评测流水线
│   └── test_adversarial.py  # 裁判区分度对抗压测脚本
├── .env.example             # 环境变量配置模板
├── .gitignore               # Git 忽略规则
└── requirements.txt         # 核心依赖清单
快速上手1. 虚拟环境配置Bashgit clone <your-repo-url>
cd llm-eval-pipeline
python -m venv venv

# Windows PowerShell 激活
.\venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
2. 配置环境变量复制环境模板并填入兼容 OpenAI 接口的大模型 API 凭证：Bashcopy .env.example .env
修改 .env 核心配置：Ini, TOMLOPENAI_API_KEY=your_actual_api_key
OPENAI_BASE_URL=[https://api.deepseek.com/v1](https://api.deepseek.com/v1)
MODEL_NAME=deepseek-chat
MOCK_MODE=false
3. 执行数据清洗管线Bashpython src/clean_pipeline.py
4. 运行自动化评测Bashpython src/eval_pipeline.py
5. 运行对抗样本区分度测试Bashpython src/test_adversarial.py
评测压测实测指标候选回答类型关联度 (Relevance)忠实度 (Faithfulness)综合判定 (Overall Pass)高质量基准回答5 / 55 / 5Pass严重事实幻觉样本2 / 51 / 5Failed答非所问/脱靶样本1 / 51 / 5Failed
