from utils.utils import load_config, get_llm, get_embedding_model
import numpy as np

# ======================================================
# Generate HyDE Answer
# ======================================================
def generate_hyde_answer(query):
    config = load_config()
    llm = get_llm(config)

    prompt = [
        ("system", "Generate a short, factual-sounding hypothetical answer to help a retrieval system. "
         "Do NOT say you are guessing; produce a confident, concise paragraph."),
        ("human", f"Question:\n{query}\n\nHypothetical Anwser:")
    ]

    synthetic_answer = llm.invoke(prompt).content.strip()
    return synthetic_answer

# ======================================================
# Generate HyDE Embeddings
# ======================================================
def generate_hyde_embedding(query):
    """
    Main HyDE function:
    1. Creates synthetic answer
    2. Converts it into embedding
    3. Returns embedding vector (np.array)
    """
    config = load_config()
    synthetic_answer = generate_hyde_answer(query)
    print(f"Query: {query}")
    print("\n🧪 Synthetic HyDE Answer: ", synthetic_answer)

    embbeding_model = get_embedding_model(config)
    return np.array(embbeding_model.embed_query(synthetic_answer), dtype = np.float32)


    

