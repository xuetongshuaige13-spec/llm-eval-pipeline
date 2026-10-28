import json
from pathlib import Path

# 确保目标路径存在（直接输出到 app.py 读取的 processed 目录）
PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 22 条 RAG 场景基准数据（符合 app.py 契约）
# ==========================================
RAG_DATA = [
    # --- A 组：黄金基准样本（全部 Pass）---
    {
        "id": "rag_001",
        "query": "香港科技大学大数据科技（Bdt）硕士的全日制学费是多少？",
        "response": "香港科技大学大数据科技硕士全日制学费为 240,000 港币，按学期分两次缴纳。",
        "contexts": [
            "香港科技大学大数据科技理学硕士（MSc in Big Data Technology）项目 2026/27 学年全日制学费为 240,000 港币，分两学期缴纳。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "检索切片精准包含了学费金额与分期细则，与问题完全对应。",
            "faithfulness_score": 5.0,
            "faithfulness_rationale": "回答严格依据文档事实，无任何外溢信息或主观推测。",
            "hallucinated_claims": [],
            "answer_relevance_score": 5.0,
            "answer_relevance_rationale": "直接清晰地回答了学费与缴纳方式，切中用户提问。",
            "overall_pass": True
        }
    },
    {
        "id": "rag_002",
        "query": "Pydantic V2 相比 V1 在性能上提升的主要底层原因是什么？",
        "response": "Pydantic V2 主要将其底层核心验证逻辑改用 Rust 编写的 pydantic-core 实现，因此执行效率相比纯 Python 的 V1 获得了 5 至 50 倍的性能提升。",
        "contexts": [
            "Pydantic V2 的核心验证逻辑用 Rust 重构为 pydantic-core，数据序列化和模式校验性能相比纯 Python 实现的 V1 提升了 5 到 50 倍。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "检索到的切片直接解释了 Rust 重构与性能对比数据。",
            "faithfulness_score": 5.0,
            "faithfulness_rationale": "回答中的 5-50 倍以及 pydantic-core 均有严格证据支持。",
            "hallucinated_claims": [],
            "answer_relevance_score": 5.0,
            "answer_relevance_rationale": "针对底层性能原因做出了结构化且准确的技术解答。",
            "overall_pass": True
        }
    },
    {
        "id": "rag_003",
        "query": "Python asyncio 中 Semaphore 的主要作用是什么？",
        "response": "asyncio.Semaphore 用于控制最大并发协程数量。它通过内部计数器限制同时执行特定任务的协程数，防止瞬时并发请求过多击垮下游 API。",
        "contexts": [
            "asyncio.Semaphore 是一个并发控制原语，内部维护一个计数器。当调用 acquire() 时计数器减一，当计数为零时后续协程被挂起等待，常用于限制对外部 API 的最大并发请求数。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "切片详细阐述了 Semaphore 原语定义及 API 场景用法。",
            "faithfulness_score": 5.0,
            "faithfulness_rationale": "完全基于切片事实总结，无无端断言。",
            "hallucinated_claims": [],
            "answer_relevance_score": 5.0,
            "answer_relevance_rationale": "准确阐明了并发控制与限流两层核心作用。",
            "overall_pass": True
        }
    },
    {
        "id": "rag_004",
        "query": "如何在 pytest 中捕获预期的异常？",
        "response": "可以使用 pytest.raises 上下文管理器，将预期会抛出异常的代码块包裹在 with pytest.raises(TargetException): 中进行断言捕获。",
        "contexts": [
            "在 pytest 中，推荐使用 pytest.raises 上下文管理器来验证代码是否如期抛出指定类型的异常，例如：with pytest.raises(ValidationError): do_something()。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "精准匹配 pytest 异常断言语法。",
            "faithfulness_score": 5.0,
            "faithfulness_rationale": "代码与语法用法均来自切片。",
            "hallucinated_claims": [],
            "answer_relevance_score": 5.0,
            "answer_relevance_rationale": "直接给出用法范式，契合提问。",
            "overall_pass": True
        }
    },
    {
        "id": "rag_005",
        "query": "Redis 的过期删除策略包含哪些？",
        "response": "Redis 同时采用惰性删除与定期删除两种机制：前者在读取键时检查过期状态，后者由后台定时周期性抽样清理过期键。",
        "contexts": [
            "Redis 结合了惰性删除（Lazy Deletion）与定期删除（Periodic Deletion）两种策略。惰性删除在访问键时检测是否过期，定期删除则每隔一段时间随机抽取部分键检查并清理。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "切片涵盖了两种删除机制的定义与区别。",
            "faithfulness_score": 5.0,
            "faithfulness_rationale": "回答没有引入上下文以外的内存淘汰策略（如 LRU/LFU），忠实度高。",
            "hallucinated_claims": [],
            "answer_relevance_score": 5.0,
            "answer_relevance_rationale": "完整罗列了两种机制并解释清晰。",
            "overall_pass": True
        }
    },
    {
        "id": "rag_006",
        "query": "HTTP 429 状态码的含义以及常用的应对策略是什么？",
        "response": "429 状态码代表请求频率超过服务端 Rate Limit。标准应对方式是遵从 Retry-After 指引，采用带随机抖动的指数退避算法进行降频重试。",
        "contexts": [
            "HTTP 429 Too Many Requests 表示客户端发送的请求过多，已超过服务端设定的速率限制。客户端常用策略是结合 Retry-After 头部进行带抖动的指数退避（Exponential Backoff with Jitter）重试。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "检索切片包含 429 含义与 Exponential Backoff with Jitter。",
            "faithfulness_score": 5.0,
            "faithfulness_rationale": "技术术语与处理机制完全符合参考切片。",
            "hallucinated_claims": [],
            "answer_relevance_score": 5.0,
            "answer_relevance_rationale": "既解释了含义又给出了标准应对策略。",
            "overall_pass": True
        }
    },
    {
        "id": "rag_007",
        "query": "什么是 RAG 三元组评估？",
        "response": "RAG 三元组评估包含三大核心指标：检索内容与问题的相关度（Context Relevance）、回答基于上下文的真实忠实度（Faithfulness）以及最终回答与用户问题的匹配度（Answer Relevance）。",
        "contexts": [
            "RAG 三元组评估指的是针对 RAG 系统核心链路的三个度量指标：Context Relevance（检索与问题的相关性）、Faithfulness（回答对检索上下文的真实忠实度）和 Answer Relevance（回答对用户意图的相关性）。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "切片完整覆盖三元组指标体系。",
            "faithfulness_score": 5.0,
            "faithfulness_rationale": "准确复述三个指标名称及其中文含义。",
            "hallucinated_claims": [],
            "answer_relevance_score": 5.0,
            "answer_relevance_rationale": "逻辑严密完整，无遗漏。",
            "overall_pass": True
        }
    },

    # --- B 组：事实性幻觉 / 忠实度缺陷（检测到细粒度原子幻觉断言）---
    {
        "id": "rag_008",
        "query": "申请香港中文大学数据科学硕士的雅思最低分数要求是多少？",
        "response": "申请该项目需要雅思总分达到 7.0 分以上，且单项不低于 6.0，同时支持六级 500 分以上直接申请。",
        "contexts": [
            "香港中文大学数据科学硕士（MSc in Data Science）对非英语母语申请者要求托福成绩不低于 79 分，或雅思（学术类）总分不低于 6.5 分。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "切片明确给出了港中文数据科学硕士的语言要求（托福 79 / 雅思 6.5）。",
            "faithfulness_score": 1.0,
            "faithfulness_rationale": "回答声称雅思需 7.0 分且支持六级，但上下文明确写明要求为 6.5 分且未支持六级，属于严重捏造事实。",
            "hallucinated_claims": [
                "申请该项目需要雅思总分达到 7.0 分以上，且单项不低于 6.0",
                "支持六级 500 分以上直接申请"
            ],
            "answer_relevance_score": 4.5,
            "answer_relevance_rationale": "回答针对了雅思要求，但由于事实错误导致可用性为 0。",
            "overall_pass": False
        }
    },
    {
        "id": "rag_009",
        "query": "某企业 AI 实习生职位的日常实习薪资待遇如何？",
        "response": "该岗位薪资标准为 400-500 元/天，并全额报销深圳往返机票与五星级酒店住宿。",
        "contexts": [
            "该大模型工程化实习生岗位日常薪资待遇为 150-200 元/天，提供免费茶歇与实习证明。"
        ],
        "eval_result": {
            "context_relevance_score": 4.5,
            "context_relevance_rationale": "检索切片包含实际岗位待遇。",
            "faithfulness_score": 1.0,
            "faithfulness_rationale": "回答捏造了 400-500 元/天的日薪以及报销机酒等不存在的福利。",
            "hallucinated_claims": [
                "该岗位薪资标准为 400-500 元/天",
                "全额报销深圳往返机票与五星级酒店住宿"
            ],
            "answer_relevance_score": 4.5,
            "answer_relevance_rationale": "形式上回答了薪资问题，但内容属于严重幻觉。",
            "overall_pass": False
        }
    },
    {
        "id": "rag_010",
        "query": "深圳北理莫斯科大学位于哪个行政区？",
        "response": "深圳北理莫斯科大学坐落于深圳市南山区科技园核心区，紧邻腾讯全球总部大楼。",
        "contexts": [
            "深圳北理莫斯科大学坐落于广东省深圳市龙岗区国际大学园路 1 号。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "切片直接标明了位于龙岗区国际大学园路 1 号。",
            "faithfulness_score": 1.0,
            "faithfulness_rationale": "回答将龙岗区篡改为南山区科技园核心区，并虚构了紧邻腾讯总部。",
            "hallucinated_claims": [
                "深圳北理莫斯科大学坐落于深圳市南山区科技园核心区",
                "紧邻腾讯全球总部大楼"
            ],
            "answer_relevance_score": 5.0,
            "answer_relevance_rationale": "回答了行政区位置，但事实完全错误。",
            "overall_pass": False
        }
    },
    {
        "id": "rag_011",
        "query": "部署 vLLM 推理框架最低需要多少张显卡？",
        "response": "部署 vLLM 推理必须具备至少 4 张 NVIDIA A100 80GB 显卡，无法在单张显卡上运行。",
        "contexts": [
            "vLLM 官方支持单卡（Single GPU）直接运行，只要显存满足模型权重与 KV Cache 分配需求即可启动服务，无需强制多卡。"
        ],
        "eval_result": {
            "context_relevance_score": 4.5,
            "context_relevance_rationale": "切片直接说明了单卡支持能力。",
            "faithfulness_score": 1.0,
            "faithfulness_rationale": "回答与切片完全对立，把单卡支持虚构成必须 4 张 A100。",
            "hallucinated_claims": [
                "部署 vLLM 推理必须具备至少 4 张 NVIDIA A100 80GB 显卡",
                "无法在单张显卡上运行"
            ],
            "answer_relevance_score": 4.5,
            "answer_relevance_rationale": "针对提问给出了明确回答，但违背参考事实。",
            "overall_pass": False
        }
    },
    {
        "id": "rag_012",
        "query": "ICAFG 2026 会议主办方是谁？",
        "response": "ICAFG 2026 由 IEEE 计算机学会与 ACM 联合主办，地点在瑞士日内瓦。",
        "contexts": [
            "2026 International Conference on Accounting, Finance and Corporate Governance (ICAFG 2026) 由 CIC 组织承办。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "切片准确包含主办方 CIC。",
            "faithfulness_score": 1.5,
            "faithfulness_rationale": "模型受先验学术会议偏置影响，误认为 IEEE 与 ACM，并虚构了地点日内瓦。",
            "hallucinated_claims": [
                "由 IEEE 计算机学会与 ACM 联合主办",
                "地点在瑞士日内瓦"
            ],
            "answer_relevance_score": 4.0,
            "answer_relevance_rationale": "回答了主办方，但内容捏造。",
            "overall_pass": False
        }
    },
    {
        "id": "rag_013",
        "query": "GitHub Actions 的每个步骤（Step）默认在同一个 Shell 环境下共享全局变量吗？",
        "response": "是的，所有步骤运行在完全同一个全局 Shell 环境中，前一步 export 的环境变量在后一步可以直接无缝读取。",
        "contexts": [
            "GitHub Actions 中同一个 Job 下的每个 Step 是独立运行在不同的子 Shell 进程中的，默认无法直接共享局部环境变量，需通过写入 GITHUB_ENV 文件进行变量持久化传递。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "切片清晰解释了 Step 的子 Shell 隔离机制与 GITHUB_ENV。",
            "faithfulness_score": 1.0,
            "faithfulness_rationale": "回答给出了与上下文完全相反的结论，断言变量可直接共享。",
            "hallucinated_claims": [
                "所有步骤运行在完全同一个全局 Shell 环境中",
                "前一步 export 的环境变量在后一步可以直接无缝读取"
            ],
            "answer_relevance_score": 5.0,
            "answer_relevance_rationale": "直接回应了是或否的问题，但技术结论完全错误。",
            "overall_pass": False
        }
    },
    {
        "id": "rag_014",
        "query": "该数据清洗管道对于空字段默认采取什么操作？",
        "response": "清洗管道遇到空字符串字段时，会自动调用 GPT-4 进行智能文本补全，并作为有效数据落盘保留。",
        "contexts": [
            "数据清洗流水线在解析到 query 或 response 为空字符串或纯空白时，将其计入 rejected_empty 计数，并在清洗结果文件中完全剔除丢弃。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "切片明确了遇到空字段执行丢弃并记录 rejected_empty 的策略。",
            "faithfulness_score": 1.0,
            "faithfulness_rationale": "回答虚构了调用 GPT-4 自动补全且落盘保留的行为。",
            "hallucinated_claims": [
                "清洗管道遇到空字符串字段时，会自动调用 GPT-4 进行智能文本补全",
                "作为有效数据落盘保留"
            ],
            "answer_relevance_score": 4.5,
            "answer_relevance_rationale": "正面回答了操作方式，但与逻辑事实严重不符。",
            "overall_pass": False
        }
    },

    # --- C 组：检索不相关 / 噪声污染（Context 质量极差）---
    {
        "id": "rag_015",
        "query": "香港留学生办理学生签证（Entry Permit）需要准备哪些材料？",
        "response": "根据提供的参考资料，未包含办理香港学生签证所需材料的信息。通常签证需要身份证件与录取通知书，但该内容未在给定文档中提及。",
        "contexts": [
            "深圳宗申赛纳维小区周边配套设施成熟，紧邻大型城市休闲绿地公园，适合居民散步慢跑。"
        ],
        "eval_result": {
            "context_relevance_score": 1.0,
            "context_relevance_rationale": "检索严重失真，检索器错误召回了小区绿化环境说明。",
            "faithfulness_score": 5.0,
            "faithfulness_rationale": "模型忠实指出参考资料未包含所需信息，未产生无证据脑补。",
            "hallucinated_claims": [],
            "answer_relevance_score": 3.0,
            "answer_relevance_rationale": "虽然做出拒答说明，但未能真正解决用户的材料咨询需求。",
            "overall_pass": False
        }
    },
    {
        "id": "rag_016",
        "query": "在 Pytest 中如何使用 fixture 共享数据库连接？",
        "response": "参考文档中没有提及 pytest fixture，仅介绍了 Docker Compose 的配置与多容器部署方式。",
        "contexts": [
            "Docker Compose 是一种用于定义和运行多容器 Docker 应用程序的工具，通过 YAML 文件配置应用服务。"
        ],
        "eval_result": {
            "context_relevance_score": 1.0,
            "context_relevance_rationale": "检索切片完全属于 Docker 领域，与 pytest 毫无关联。",
            "faithfulness_score": 5.0,
            "faithfulness_rationale": "忠实陈述文档局限，无虚假内容。",
            "hallucinated_claims": [],
            "answer_relevance_score": 3.0,
            "answer_relevance_rationale": "受限于无关上下文只能拒答。",
            "overall_pass": False
        }
    },
    {
        "id": "rag_017",
        "query": "如何使用 Pandas 对 DataFrame 某一列的缺失值进行中位数填充？",
        "response": "提供的参考文本为网络游戏介绍，未包含 Pandas 填充缺失值的方法。请检查检索源配置。",
        "contexts": [
            "英雄联盟是一款由 Riot Games 开发的 MOBA 竞技 PC 游戏，玩家在召唤师峡谷中进行 5v5 团队对战。"
        ],
        "eval_result": {
            "context_relevance_score": 1.0,
            "context_relevance_rationale": "检索切片为游戏百科，发生灾难性检索偏差。",
            "faithfulness_score": 5.0,
            "faithfulness_rationale": "未强行用游戏切片编造 Pandas 代码。",
            "hallucinated_claims": [],
            "answer_relevance_score": 2.5,
            "answer_relevance_rationale": "无法给出有效的代码解答。",
            "overall_pass": False
        }
    },
    {
        "id": "rag_018",
        "query": "大模型量化中 INT8 与 FP16 相比能节省多少显存？",
        "response": "参考资料仅为成都商业中心设施介绍，缺乏大模型 INT8/FP16 显存开销的相关数据。",
        "contexts": [
            "成都来福士广场集购物中心、办公楼与特色餐饮于一体，深受年轻群体与养宠人士喜爱。"
        ],
        "eval_result": {
            "context_relevance_score": 1.0,
            "context_relevance_rationale": "召回了商业地产简介，与模型量化无关。",
            "faithfulness_score": 5.0,
            "faithfulness_rationale": "模型选择如实拒答。",
            "hallucinated_claims": [],
            "answer_relevance_score": 2.5,
            "answer_relevance_rationale": "未能回答量化显存差值。",
            "overall_pass": False
        }
    },
    {
        "id": "rag_019",
        "query": "Tenacity 库中 stop_after_attempt(3) 的具体作用是什么？",
        "response": "文档内容为明朝开国皇帝历史生平，无法从中获取 tenacity 库的函数用法说明。",
        "contexts": [
            "明太祖朱元璋生于濠州钟离，建立了大明王朝，并强化了中央集权制度。"
        ],
        "eval_result": {
            "context_relevance_score": 1.0,
            "context_relevance_rationale": "召回历史人物生平，与 Python 重试库无关。",
            "faithfulness_score": 5.0,
            "faithfulness_rationale": "正确识别出文档内容不匹配并如实指出。",
            "hallucinated_claims": [],
            "answer_relevance_score": 2.5,
            "answer_relevance_rationale": "拒答未能直接解决重试语法问题。",
            "overall_pass": False
        }
    },

    # --- D 组：回答跑题 / 语义偏离（Answer Relevance 极低）---
    {
        "id": "rag_020",
        "query": "请给出当前项目中计算准确率的具体数学公式是什么？",
        "response": "评测是现代机器学习工程非常重要的环节。我们必须重视测试驱动开发，做好团队代码质量管理，才能保证系统的稳定交付。",
        "contexts": [
            "在评测体系中，分类准确率定义为正确分类的样本数与总样本数的比值，公式为 Accuracy = (TP + TN) / (TP + TN + FP + FN)。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "切片给出了精准的准确率混淆矩阵公式。",
            "faithfulness_score": 4.0,
            "faithfulness_rationale": "回答说的大致不错但未体现切片核心公式。",
            "hallucinated_claims": [],
            "answer_relevance_score": 1.5,
            "answer_relevance_rationale": "严重跑题，用户索要具体公式，回答空谈敏捷与质量管理。",
            "overall_pass": False
        }
    },
    {
        "id": "rag_021",
        "query": "Streamlit 中 st.metric 组件由哪三个主要参数组成？",
        "response": "Streamlit 是非常优秀的前端展示框架，可以用很少的代码搭建数据分析 Web 看板，非常适合 Python 数据工程师使用。",
        "contexts": [
            "st.metric 的核心输入参数包括 label（指标标签）、value（当前数值）以及可选的 delta（变化量或环比差值）。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "切片详细罗列了 label, value, delta 三大参数。",
            "faithfulness_score": 4.0,
            "faithfulness_rationale": "内容泛泛而谈，未违背大事实。",
            "hallucinated_claims": [],
            "answer_relevance_score": 1.0,
            "answer_relevance_rationale": "答非所问，完全未回答核心参数提问。",
            "overall_pass": False
        }
    },
    {
        "id": "rag_022",
        "query": "Git 命令中 git reset --hard 和 git reset --soft 的本质区别是什么？",
        "response": "在写 Git 提交记录时，我们应当遵循规范，使用 feat、fix、docs 等前缀保持提交历史清爽。",
        "contexts": [
            "git reset --soft 会保留工作区与暂存区的所有修改，仅移动 HEAD 指针；而 --hard 会彻底丢弃暂存区与工作区的所有未提交改动。"
        ],
        "eval_result": {
            "context_relevance_score": 5.0,
            "context_relevance_rationale": "切片精准区分了 soft 与 hard 对工作区暂存区的影响。",
            "faithfulness_score": 4.0,
            "faithfulness_rationale": "表述本身无害但偏离主题。",
            "hallucinated_claims": [],
            "answer_relevance_score": 1.0,
            "answer_relevance_rationale": "用户问指针回滚差异，模型却讲 Commit 提交规范，完全偏题。",
            "overall_pass": False
        }
    }
]


def main():
    # 直接写入 Streamlit app.py 读取的目标文件
    target_path = PROCESSED_DIR / "rag_eval_results.jsonl"
    with open(target_path, "w", encoding="utf-8") as f:
        for item in RAG_DATA:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ 成功写入 22 条完整契约的 RAG 评测基准数据: {target_path}")

    # 同时同步一份到 raw 目录备份
    backup_path = RAW_DIR / "rag_benchmark_30.jsonl"
    with open(backup_path, "w", encoding="utf-8") as f:
        for item in RAG_DATA:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ 同步备份至: {backup_path}")


if __name__ == "__main__":
    main()
