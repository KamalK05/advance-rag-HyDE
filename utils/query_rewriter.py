from utils.utils import load_config, get_llm


def rewrite_query(user_query: str, config):
    """
    Rewrites or expands a query to be more specific, contextual, and
    retrieval-friendly. Works with OpenAI or Gemini.
    """
    llm = get_llm(config)

    system_prompt = """
You are an expert at improving user queries for retrieval-based search.
IMPORTANT
- include all required technical terms
- Rewrite the query to make it clearer, longer, and more specific, while keeping the original intent.
"""

    prompt = [
        ("system", system_prompt),
        ("human", f"Rewrite the following query:\n\n{user_query}")
    ]

    rewritten_query = llm.invoke(prompt)
    return rewritten_query.content.strip()