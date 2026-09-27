# utils/retriever.py
import os
from langchain_community.vectorstores import FAISS
from utils.utils import load_config, get_embedding_model

# ======================================================
# Get Index
# ======================================================
def get_index_path(provider, config):
    if provider == "openai":
        index_path = config["vectordb"]["faiss_openai"]
    else:
       index_path = config["vectordb"]["faiss_gemini"]

    return index_path

# ======================================================
# Create FAISS Retriever
# ======================================================
def create_retriever(provider, chunks, config):
    index_path = get_index_path(provider, config)
    enbedding_model = get_embedding_model(config)

    os.makedirs(index_path, exist_ok = True)

    vectorstore = FAISS.from_documents(chunks, embedding = enbedding_model)
    vectorstore.save_local(index_path)

    print(f"✅ Created new FAISS index ({provider}) at: {index_path}")

    return vectorstore.as_retriever(search_kwargs = {"k": config["retrieval"].get("top_k", 3)})

# ======================================================
# Load FAISS Retriever
# ======================================================
def load_retriever(provider, config):
    index_path = get_index_path(provider, config)
    embedding_model = get_embedding_model(config)
    vectorstore = FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization = True
    )

    print(f"✅ Loaded FAISS index ({provider}) from: {index_path}")
    return vectorstore.as_retriever(search_kwargs = {"k": config["retrieval"].get("top_k", 3)})


# ======================================================
# Get FAISS Retriever
# ======================================================
def get_retriever(config, chunks_if_needed = None):
    provider = config["llm"]["provider"]
    index_path = get_index_path(provider, config)

    if not os.path.exists(index_path):
        print(f"⚠️ No FAISS index found for provider '{provider}'. Creating one...\n")

        if chunks_if_needed is None:
            raise RuntimeError(
                "Chunks not provided, The caller must supply the chunks when index doesn`t exist."
            )    
        
        return create_retriever(provider, chunks_if_needed, config)
    else:
        return load_retriever(provider, config)
