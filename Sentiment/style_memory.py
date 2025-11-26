import os
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


COLLECTION_NAME = "sentiment_style"


def _get_persist_dir() -> str:
    base = os.path.join(os.path.dirname(__file__), "vector_store", "sentiment")
    os.makedirs(base, exist_ok=True)
    return base


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. Please set it before running."
        )
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)


def _get_vectorstore() -> Chroma:
    embeddings = _get_embeddings()
    persist_dir = _get_persist_dir()
    # This will create or load the collection
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


def upsert_style_guide() -> None:
    """Seed or refresh the style guide and exemplar docs in the vector store."""
    vs = _get_vectorstore()

    core_rules = (
        "Role: HR analytics assistant. Start every response with 'Of course. As an HR analytics assistant,'. "
        "Keep professional HR tone. Prioritize clarity, evidence, and actionable steps."
    )
    structure = (
        "Structure: Exactly the following headings in order: '1. Sentiment Analysis', '2. Summary of Employee Opinion', "
        "'Key Positives (What's Working)', 'Key Areas for Improvement / Attrition Risks'. Use bullets where implied."
    )
    constraints = (
        "Constraints: 450-500 words. Sentiment percentages sum to 100%. Reference actual survey content. "
        "Map each problem to a concrete retention strategy from the provided list."
    )
    exemplar = (
        "Exemplar: Maintain concise bullets, avoid generic claims, cite survey phrases. "
        "Limit to max two sentences per attrition factor's problem and suggestion."
    )

    texts: List[str] = [core_rules, structure, constraints, exemplar]
    ids: List[str] = [
        "style_core_rules_v1",
        "style_structure_v1",
        "style_constraints_v1",
        "style_exemplar_v1",
    ]

    try:
        # Delete then re-add to ensure latest
        vs.delete(ids=ids)
    except Exception:
        pass

    vs.add_texts(texts=texts, ids=ids, metadatas=[{"type": "style"} for _ in texts])
    vs.persist()


def get_style_context(query: str, k: int = 3) -> str:
    vs = _get_vectorstore()
    docs = vs.similarity_search(query=query, k=k)
    joined = "\n\n".join(d.page_content for d in docs)
    return joined


def save_output_example(text: str) -> None:
    if not text or not text.strip():
        return
    vs = _get_vectorstore()
    # Tag as generated to allow filtering later
    vs.add_texts([text], metadatas=[{"type": "generated_example"}])
    vs.persist()
