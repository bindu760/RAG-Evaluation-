
def get_rag_answer(question, vector_store, llm, k=3):
    """
    Retrieve relevant documents and generate
    an answer using the LLM.
    """

    # Step 1: Retrieve relevant documents
    retrieved_docs = vector_store.similarity_search(
        question,
        k=k
    )

    # Step 2: Combine retrieved document contents
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Step 3: Create prompt
    prompt = f"""
    You are a helpful AI assistant.

    Answer the question using only the provided context.

    If the answer is not available in the context,
    say that you don't know based on the provided documents.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Step 4: Generate answer using Groq
    response = llm.invoke(prompt)

    # Step 5: Return answer and retrieved documents
    return {
        "answer": response.content,
        "retrieved_documents": retrieved_docs
    }
