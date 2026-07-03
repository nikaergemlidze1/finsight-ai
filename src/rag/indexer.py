import yaml
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()


def _load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_index(config_path: str = "config/config.yaml") -> None:
    cfg         = _load_config(config_path)
    data_dir    = cfg["rag"]["docs_dir"]
    persist_dir = cfg["rag"]["index_persist_dir"]

    Settings.llm = OpenAI(model=cfg["rag"]["llm_model"])
    Settings.embed_model = HuggingFaceEmbedding(model_name=cfg["rag"]["embed_model"])

    print(f"Reading documents from {data_dir} ...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"Loaded {len(documents)} document(s). Building vector index ...")

    index = VectorStoreIndex.from_documents(documents)

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Index saved to {persist_dir}")


if __name__ == "__main__":
    build_index()