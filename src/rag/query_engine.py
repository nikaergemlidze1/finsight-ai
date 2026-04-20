import yaml
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()


def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_query_engine(config_path: str = "config/config.yaml"):
    """Load the persisted LlamaIndex vector store and return a query engine."""
    cfg         = _load_config(config_path)
    persist_dir = cfg["rag"]["index_persist_dir"]

    if not Path(persist_dir).exists():
        raise FileNotFoundError(
            f"No RAG index found at '{persist_dir}'. "
            "Run `python -m src.rag.indexer` first."
        )

    Settings.llm = OpenAI(model=cfg["rag"]["llm_model"])
    Settings.embed_model = HuggingFaceEmbedding(model_name=cfg["rag"]["embed_model"])

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index.as_query_engine(similarity_top_k=cfg["rag"]["similarity_top_k"])


if __name__ == "__main__":
    engine = get_query_engine()
    question = "What is our strategy for targeting retirees?"
    print(f"Q: {question}\nA: {engine.query(question)}")