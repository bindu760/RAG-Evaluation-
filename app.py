
import streamlit as st
import pandas as pd
from pathlib import Path


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# 2. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

retrieval_file = RESULTS_DIR / "retrieval_evaluation.csv"
generation_file = RESULTS_DIR / "generation_evaluation.csv"
top_k_file = RESULTS_DIR / "top_k_comparison.csv"


# ==========================================
# 3. LOAD CSV FILES
# ==========================================

def load_csv(file_path):

    if file_path.exists():
        return pd.read_csv(file_path)

    return pd.DataFrame()


retrieval_df = load_csv(retrieval_file)
generation_df = load_csv(generation_file)
top_k_df = load_csv(top_k_file)


# ==========================================
# 4. DASHBOARD HEADER
# ==========================================

st.title("📊 RAG Evaluation & Benchmarking System")

st.write(
    "Evaluate and benchmark Retrieval-Augmented Generation systems."
)


# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Select a page",
    [
        "Overview",
        "Retrieval Evaluation",
        "Generation Evaluation",
        "Top-K Comparison",
        "Reports"
    ]
)


# ==========================================
# 6. OVERVIEW PAGE
# ==========================================

if page == "Overview":

    st.header("Dashboard Overview")

    total_questions = len(generation_df)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Evaluation Questions",
        total_questions
    )

    col2.metric(
        "Retrieval Metrics",
        4
    )

    col3.metric(
        "Generation Metrics",
        3
    )

    st.success("Dashboard loaded successfully!")

    st.subheader("Average Evaluation Scores")

    if not retrieval_df.empty and not generation_df.empty:

        retrieval_avg = retrieval_df[
            ["precision", "recall", "hit_rate", "mrr"]
        ].mean()

        generation_avg = generation_df[
            [
                "faithfulness",
                "answer_relevancy",
                "answer_correctness"
            ]
        ].mean()

        col1, col2 = st.columns(2)

        with col1:

            st.write("Retrieval Metrics")

            st.dataframe(
                retrieval_avg.to_frame("Score"),
                use_container_width=True
            )

        with col2:

            st.write("Generation Metrics")

            st.dataframe(
                generation_avg.to_frame("Score"),
                use_container_width=True
            )

    else:

        st.warning("Evaluation CSV files are missing.")


# ==========================================
# 7. RETRIEVAL EVALUATION PAGE
# ==========================================

elif page == "Retrieval Evaluation":

    st.header("📊 Retrieval Evaluation")

    if not retrieval_df.empty:

        retrieval_avg = retrieval_df[
            ["precision", "recall", "hit_rate", "mrr"]
        ].mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Precision@K",
            f"{retrieval_avg['precision']:.2%}"
        )

        col2.metric(
            "Recall@K",
            f"{retrieval_avg['recall']:.2%}"
        )

        col3.metric(
            "Hit Rate@K",
            f"{retrieval_avg['hit_rate']:.2%}"
        )

        col4.metric(
            "MRR@K",
            f"{retrieval_avg['mrr']:.2%}"
        )

        st.divider()

        st.subheader("Average Retrieval Performance")

        st.bar_chart(retrieval_avg)

        st.subheader("Question-wise Evaluation Results")

        st.dataframe(
            retrieval_df,
            use_container_width=True
        )

    else:

        st.warning("Retrieval results not found.")


# ==========================================
# 8. GENERATION EVALUATION PAGE
# ==========================================

elif page == "Generation Evaluation":

    st.header("🤖 Generation Evaluation")

    if not generation_df.empty:

        generation_avg = generation_df[
            [
                "faithfulness",
                "answer_relevancy",
                "answer_correctness"
            ]
        ].mean()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Faithfulness",
            f"{generation_avg['faithfulness']:.2%}"
        )

        col2.metric(
            "Answer Relevancy",
            f"{generation_avg['answer_relevancy']:.2%}"
        )

        col3.metric(
            "Answer Correctness",
            f"{generation_avg['answer_correctness']:.2%}"
        )

        st.divider()

        st.subheader("Average Generation Performance")

        st.bar_chart(generation_avg)

        st.subheader("Question-wise Evaluation Results")

        st.dataframe(
            generation_df,
            use_container_width=True
        )

    else:

        st.warning("Generation results not found.")


# ==========================================
# 9. TOP-K COMPARISON PAGE
# ==========================================

elif page == "Top-K Comparison":

    st.header("📈 Top-K Comparison")

    if not top_k_df.empty:

        st.subheader("Retrieval Performance at Different K Values")

        st.dataframe(
            top_k_df,
            use_container_width=True
        )

        st.divider()

        st.subheader("Precision Comparison")

        st.line_chart(
            top_k_df.set_index("top_k")[["precision"]]
        )

        st.subheader("Recall Comparison")

        st.line_chart(
            top_k_df.set_index("top_k")[["recall"]]
        )

        st.subheader("All Retrieval Metrics")

        chart_data = top_k_df.set_index("top_k")[
            [
                "precision",
                "recall",
                "hit_rate",
                "mrr"
            ]
        ]

        st.line_chart(chart_data)

    else:

        st.warning(
            "Top-K results not found. "
            "Run evaluation/save_top_k_results.py first."
        )


# ==========================================
# 10. REPORTS PAGE
# ==========================================

elif page == "Reports":

    st.header("📄 Evaluation Reports")

    if not retrieval_df.empty:

        st.subheader("Retrieval Evaluation Report")

        st.download_button(
            label="Download Retrieval Results",
            data=retrieval_df.to_csv(index=False),
            file_name="retrieval_evaluation.csv",
            mime="text/csv"
        )

    if not generation_df.empty:

        st.subheader("Generation Evaluation Report")

        st.download_button(
            label="Download Generation Results",
            data=generation_df.to_csv(index=False),
            file_name="generation_evaluation.csv",
            mime="text/csv"
        )

    if not top_k_df.empty:

        st.subheader("Top-K Comparison Report")

        st.download_button(
            label="Download Top-K Results",
            data=top_k_df.to_csv(index=False),
            file_name="top_k_comparison.csv",
            mime="text/csv"
        )

    if (
        retrieval_df.empty
        and generation_df.empty
        and top_k_df.empty
    ):

        st.warning("No evaluation reports available.")
