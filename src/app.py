import json
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ================================
# 页面全局配置
# ================================
st.set_page_config(
    page_title="LLM & RAG 评测与幻觉审计看板",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 工程目录定位
BASE_DIR = Path(__file__).resolve().parent.parent
RAG_RESULT_PATH = BASE_DIR / "data" / "processed" / "rag_eval_results.jsonl"
QA_RESULT_PATH = BASE_DIR / "data" / "processed" / "eval_results.jsonl"


# ================================
# 数据加载与缓存
# ================================
@st.cache_data
def load_rag_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            eval_res = item.get("eval_result", {})
            records.append({
                "id": item.get("id"),
                "query": item.get("query"),
                "response": item.get("response"),
                "contexts": item.get("contexts", []),
                "context_relevance": eval_res.get("context_relevance_score", 0),
                "context_rationale": eval_res.get("context_relevance_rationale", ""),
                "faithfulness": eval_res.get("faithfulness_score", 0),
                "faithfulness_rationale": eval_res.get("faithfulness_rationale", ""),
                "hallucinated_claims": eval_res.get("hallucinated_claims", []),
                "answer_relevance": eval_res.get("answer_relevance_score", 0),
                "answer_rationale": eval_res.get("answer_relevance_rationale", ""),
                "overall_pass": eval_res.get("overall_pass", False),
            })
    return pd.DataFrame(records)


# ================================
# 侧边栏导航
# ================================
st.sidebar.title("🛠️ 评测管线控制台")
st.sidebar.caption("LLM Data Clean & Eval Pipeline")

view_mode = st.sidebar.radio(
    "选择看板视图",
    ["📊 RAG 三元组与幻觉审计", "📋 基础 QA 自动化评测"],
    index=0
)

st.sidebar.divider()
st.sidebar.markdown("""
**核心考察维度说明**：
- **Context Relevance**: 检索切片命中率
- **Faithfulness**: 基于事实生成无幻觉
- **Answer Relevance**: 对齐用户提问意图
""")

# ================================
# 视图 1：RAG 三元组与幻觉审计
# ================================
if view_mode == "📊 RAG 三元组与幻觉审计":
    st.title("🎯 RAG 三元组评测与细粒度幻觉分析看板")
    st.caption("基于 LLM-as-a-Judge 与原子断言（Atomic Claims）检测机制构建")

    df = load_rag_data(RAG_RESULT_PATH)

    if df.empty:
        st.warning(f"⚠️ 未检测到评测结果文件: `{RAG_RESULT_PATH}`。请先运行 `python src/rag_eval_pipeline.py` 生成评测数据。")
        st.stop()

    # 1. KPI 概览指标卡
    total_count = len(df)
    pass_count = df["overall_pass"].sum()
    pass_rate = (pass_count / total_count) * 100 if total_count > 0 else 0
    avg_ctx = df["context_relevance"].mean()
    avg_faith = df["faithfulness"].mean()
    avg_ans = df["answer_relevance"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("总评测样本数", f"{total_count} 条")
    col2.metric("综合合格率", f"{pass_rate:.1f}%", delta=f"{pass_count}/{total_count} Pass")
    col3.metric("检索相关度", f"{avg_ctx:.2f} / 5.0")
    col4.metric("回答忠实度", f"{avg_faith:.2f} / 5.0", delta_color="inverse" if avg_faith < 3.0 else "normal")
    col5.metric("回答相关度", f"{avg_ans:.2f} / 5.0")

    st.divider()

    # 2. 图表联动区域
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.subheader("📈 三元组指标均值对比")
        radar_df = pd.DataFrame({
            "维度": ["上下文相关度", "回答忠实度", "回答相关度"],
            "得分": [avg_ctx, avg_faith, avg_ans],
            "基准通过线": [3.0, 3.0, 3.0]
        })
        fig_bar = px.bar(
            radar_df, x="维度", y="得分", text="得分",
            color="维度", range_y=[0, 5.5],
            color_discrete_sequence=["#3366CC", "#DC3912", "#FF9900"]
        )
        fig_bar.add_hline(y=3.0, line_dash="dash", line_color="gray", annotation_text="及格线 (3.0)")
        fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_bar.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        st.subheader("🥧 综合合格率分布")
        status_counts = df["overall_pass"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        status_counts["status"] = status_counts["status"].map({True: "合格 (Pass)", False: "未达标 (Failed)"})
        fig_pie = px.pie(
            status_counts, names="status", values="count",
            color="status",
            color_discrete_map={"合格 (Pass)": "#2CA02C", "未达标 (Failed)": "#D62728"},
            hole=0.45
        )
        fig_pie.update_layout(height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # 3. 样本下钻与幻觉归因详情
    st.subheader("🔬 样本明细与幻觉追踪")

    # 筛选控件
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        filter_status = st.selectbox("筛选状态", ["全部样本", "仅看未通过 (Failed)", "仅看存在幻觉样本"])

    filtered_df = df.copy()
    if filter_status == "仅看未通过 (Failed)":
        filtered_df = filtered_df[~filtered_df["overall_pass"]]
    elif filter_status == "仅看存在幻觉样本":
        filtered_df = filtered_df[filtered_df["hallucinated_claims"].apply(lambda x: len(x) > 0)]

    # 遍历展示样本卡片
    for _, row in filtered_df.iterrows():
        badge = "🟢 PASS" if row["overall_pass"] else "🔴 FAILED"
        has_hallucination = len(row["hallucinated_claims"]) > 0

        with st.expander(f"【{badge}】样本 [{row['id']}] : {row['query']}", expanded=not row["overall_pass"]):
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("检索相关度 (Context)", f"{row['context_relevance']} / 5")
            sc2.metric("回答忠实度 (Faithfulness)", f"{row['faithfulness']} / 5")
            sc3.metric("回答相关度 (Answer Rel)", f"{row['answer_relevance']} / 5")

            st.markdown("#### 1. 检索召回切片 (Retrieved Contexts)")
            for i, ctx in enumerate(row["contexts"], 1):
                st.info(f"**切片 {i}**: {ctx}")

            st.markdown("#### 2. 模型生成回答 (Response)")
            st.code(row["response"], language=None)

            if has_hallucination:
                st.error("🚨 **检测到原子级事实幻觉断言（未被上下文支撑）**:")
                for claim in row["hallucinated_claims"]:
                    st.markdown(f"- ❌ `{claim}`")

            st.markdown("#### 3. 裁判模型判定推理 (Rationales)")
            st.markdown(f"- **检索评分归因**: {row['context_rationale']}")
            st.markdown(f"- **忠实度归因**: {row['faithfulness_rationale']}")
            st.markdown(f"- **回答相关度归因**: {row['answer_rationale']}")

# ================================
# 视图 2：基础 QA 评测看板
# ================================
else:
    st.title("📋 基础问答与对抗评测看板")
    if not QA_RESULT_PATH.exists():
        st.warning(f"⚠️ 未检测到评测数据: `{QA_RESULT_PATH}`，请先运行 `python src/eval_pipeline.py`。")
    else:
        qa_records = []
        with open(QA_RESULT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    judge = item.get("judge", {})
                    qa_records.append({
                        "ID": item.get("id"),
                        "问题": item.get("query"),
                        "相关度得分": judge.get("relevance_score"),
                        "忠实度得分": judge.get("faithfulness_score"),
                        "判定结果": "Pass" if judge.get("overall_pass") else "Fail",
                        "关联度归因": judge.get("relevance_rationale"),
                        "忠实度归因": judge.get("faithfulness_rationale"),
                    })
        qa_df = pd.DataFrame(qa_records)
        st.dataframe(qa_df, use_container_width=True)
