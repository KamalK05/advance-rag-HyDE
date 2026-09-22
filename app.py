# app.py
from utils.utils import load_config, get_llm
from utils.loader import load_and_chunk_docs
from utils.retriever import get_retriever
from utils.query_rewriter import rewrite_query
from utils.hyde_generator import generate_hyde_embedding

# ======================================================
# RAG Pipelines
# ======================================================
# --- Baseline retrieval ---
def run_baseline(query, retriever, llm):
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = [
        ("system", "Answer the question ONLY using the context provided."),
        ("human", f"Context:\n{context}\n\nQuestion:\n{query}")
    ]
    response = llm.invoke(prompt).content.strip()
    return response

# --- Query Rewriting retrieval ---
def run_rewritten(query, retriever, llm, config):
    rewritten_query = rewrite_query(query, config)
    print(f"\n🔁 Rewritten Query → {rewritten_query}")

    docs = retriever.invoke(rewritten_query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = [
        ("system", "Use only the retrieved context to answer the question."),
        ("human", f"Context:\n{context}\n\nQuestion:\n{rewritten_query}\n\nAnswer:")
    ]

    response = llm.invoke(prompt).content.strip()
    return response

# --- HyDE retrieval ---
def run_hyde(query, retriever, llm):
    hyde_embeddings = generate_hyde_embedding(query)

    docs = retriever.vectorstore.similarity_search_by_vector(hyde_embeddings, k = 5)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = [
        ("system", "Use only the retrieved context to answer the question."),
        ("human", f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:")
    ]

    response = llm.invoke(prompt).content.strip()
    return response

# ======================================================
# MAIN PROGRAM
# ======================================================
def main():
    # ======================================================
    # Load config + Get correct LLM (OpenAI | Gemini)
    # ======================================================
    config = load_config()
    llm = get_llm(config)

    # ======================================================
    # Build or load FAISS based on provider
    # ======================================================
    chunks = load_and_chunk_docs("./data/raw/insurance_docs", chunk_size = 50, chunk_overlap = 20)
    retriever = get_retriever(config, chunks_if_needed = chunks)

    # ======================================================
    # User Query
    # ======================================================
    query = input("\n🔍 Please enter your query: ").strip()

    # ======================================================
    # BASELINE RAG
    # ======================================================
    print("\n====================")
    print("✅ BASELINE RAG")
    print("======================")

    baseline_response = run_baseline(query, retriever, llm)
    print(f"Query - {query}")
    print(f"Baseline Answer - {baseline_response}")
    print("-" * 50)


    # ======================================================
    # QUERY REWRITING RAG
    # ======================================================
    print("\n==========================")
    print("✅ QUERY REWRITING + RAG")
    print("============================")

    rewritting_response = run_rewritten(query, retriever, llm, config)
    print(f"Query Rewritting Answer - {rewritting_response}")
    print("-" * 50)

    # ======================================================
    # HYDE RAG
    # ======================================================
    print("\n====================")
    print("✅ HYDE RAG")
    print("======================")

    hyde_response = run_hyde(query, retriever, llm)
    print(f"\n🤖hyde + retriever Answer - {hyde_response}")

if __name__ == "__main__":
    main()

