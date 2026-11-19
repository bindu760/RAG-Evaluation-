
import streamlit as st
import pandas as pd
import tempfile

from pathlib import Path

from src.rag_pipeline import (
    build_rag_pipeline,
    ask_pdf_question
)


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
performance_file = RESULTS_DIR / "performance_evaluation.csv"


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
performance_df = load_csv(performance_file)


# Remove average row for performance calculations
if not performance_df.empty:

    performance_detail_df = performance_df[
        performance_df["question"] != "AVERAGE"
    ].copy()

else:

    performance_detail_df = pd.DataFrame()


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
        "RAG Playground",
        "Retrieval Evaluation",
        "Generation Evaluation",
        "Top-K Comparison",
        "Performance Evaluation",
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

    col1.metric("Evaluation Questions", total_questions)
    col2.metric("Retrieval Metrics", 4)
    col3.metric("Generation Metrics", 3)

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
# 7. RAG PLAYGROUND PAGE
# ==========================================

elif page == "RAG Playground":

    st.header("📄 RAG Playground")

    st.write(
        "Upload a PDF document and ask questions about its content."
    )

    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "llm" not in st.session_state:
        st.session_state.llm = None

    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # PDF uploader
    uploaded_file = st.file_uploader(
        "Upload your PDF document",
        type=["pdf"]
    )

    # Process PDF
    if uploaded_file is not None:

        if st.session_state.pdf_name != uploaded_file.name:

            with st.spinner("Processing PDF..."):

                temp_path = None

                try:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    vector_store, llm, chunk_count = build_rag_pipeline(
                        temp_path
                    )

                    st.session_state.vector_store = vector_store
                    st.session_state.llm = llm
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []

                    st.success(
                        f"PDF processed successfully! "
                        f"Created {chunk_count} chunks."
                    )

                except Exception as error:

                    st.session_state.vector_store = None
                    st.session_state.llm = None
                    st.session_state.pdf_name = None

                    st.error(f"PDF processing failed: {error}")

                finally:

                    if temp_path and Path(temp_path).exists():
                        Path(temp_path).unlink()

        else:

            st.success(
                f"Ready to ask questions about {uploaded_file.name}"
            )

    # Ask questions
    if st.session_state.vector_store is not None:

        st.divider()

        st.subheader("Ask Questions About Your PDF")

        question = st.text_input(
            "Enter your question",
            key="pdf_question"
        )

        if st.button("Generate Answer"):

            if question.strip():

                with st.spinner("Generating answer..."):

                    try:

                        result = ask_pdf_question(
                            question,
                            st.session_state.vector_store,
                            st.session_state.llm,
                            k=3
                        )

                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": result["answer"],
                            "latency": result["latency"],
                            "input_tokens": result["input_tokens"],
                            "output_tokens": result["output_tokens"],
                            "retrieved_documents": result["retrieved_documents"]
                        })

                    except Exception as error:

                        st.error(
                            f"Question answering failed: {error}"
                        )

            else:

                st.warning("Please enter a question.")

        # Display chat history
        for chat in st.session_state.chat_history:

            st.markdown("**You:**")
            st.write(chat["question"])

            st.markdown("**AI Assistant:**")
            st.write(chat["answer"])

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Latency",
                f"{chat['latency']:.2f} sec"
            )

            col2.metric(
                "Input Tokens",
                chat["input_tokens"]
            )

            col3.metric(
                "Output Tokens",
                chat["output_tokens"]
            )

            with st.expander("View Retrieved Context"):

                for index, doc in enumerate(
                    chat["retrieved_documents"],
                    start=1
                ):

                    st.write(f"Document {index}")
                    st.write(doc.page_content)

                    st.caption(
                        f"Source: {doc.metadata.get('source', 'Unknown')}"
                    )

            st.divider()

        if st.button("Clear Chat History"):

            st.session_state.chat_history = []
            st.rerun()

    else:

        st.info("Upload a PDF to start asking questions.")


# ==========================================
# 8. RETRIEVAL EVALUATION PAGE
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
# 9. GENERATION EVALUATION PAGE
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
# 10. TOP-K COMPARISON PAGE
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

        chart_data = top_k_df.set_index("top_k")[
            ["precision", "recall", "hit_rate", "mrr"]
        ]

        st.subheader("Top-K Retrieval Metrics")

        st.line_chart(chart_data)

    else:

        st.warning(
            "Top-K results not found. Run save_top_k_results.py first."
        )


# ==========================================
# 11. PERFORMANCE EVALUATION PAGE
# ==========================================

elif page == "Performance Evaluation":

    st.header("⚡ Performance Evaluation")

    if not performance_detail_df.empty:

        avg_latency = performance_detail_df["latency"].mean()

        avg_input_tokens = performance_detail_df["input_tokens"].mean()

        avg_output_tokens = performance_detail_df["output_tokens"].mean()

        avg_total_tokens = performance_detail_df["total_tokens"].mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Average Latency",
            f"{avg_latency:.2f} sec"
        )

        col2.metric(
            "Average Input Tokens",
            f"{avg_input_tokens:.0f}"
        )

        col3.metric(
            "Average Output Tokens",
            f"{avg_output_tokens:.0f}"
        )

        col4.metric(
            "Average Total Tokens",
            f"{avg_total_tokens:.0f}"
        )

        st.divider()

        st.subheader("Question-wise Response Latency")

        latency_chart = performance_detail_df.set_index(
            "question"
        )[["latency"]]

        st.bar_chart(latency_chart)

        st.subheader("Token Usage Comparison")

        token_chart = performance_detail_df.set_index(
            "question"
        )[
            ["input_tokens", "output_tokens", "total_tokens"]
        ]

        st.bar_chart(token_chart)

        st.subheader("Detailed Performance Results")

        st.dataframe(
            performance_df,
            use_container_width=True
        )

    else:

        st.warning(
            "Performance results not found. "
            "Run save_performance_results.py first."
        )


# ==========================================
# 12. REPORTS PAGE
# ==========================================

elif page == "Reports":

    st.header("📄 Evaluation Reports")

    reports = [
        (
            "Retrieval Results",
            retrieval_df,
            "retrieval_evaluation.csv"
        ),
        (
            "Generation Results",
            generation_df,
            "generation_evaluation.csv"
        ),
        (
            "Top-K Results",
            top_k_df,
            "top_k_comparison.csv"
        ),
        (
            "Performance Results",
            performance_df,
            "performance_evaluation.csv"
        )
    ]

    for title, dataframe, filename in reports:

        if not dataframe.empty:

            st.subheader(title)

            st.download_button(
                label=f"Download {title}",
                data=dataframe.to_csv(index=False),
                file_name=filename,
                mime="text/csv",
                key=filename
            )

    if all(
        dataframe.empty
        for _, dataframe, _ in reports
    ):

        st.warning("No evaluation reports available.")
