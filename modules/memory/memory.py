from doctest import debug
import uuid
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from typing import List, Any
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# print onlky if debug is True 
debug = True

def print_debug(message):
    if debug:
        print(message)

class Memory:
    def __init__(self, db_dir: str = "./memory_db"):
        self.db_dir = db_dir
        self.semantic_memory = Chroma(
            collection_name="semantic_memory",
            embedding_function=embeddings,
            persist_directory=self.db_dir,
        )
        self.episodic_memory = Chroma(
            collection_name="episodic_memory",
            embedding_function=embeddings,
            persist_directory=self.db_dir,
        )
        self.procedural_memory = Chroma(
            collection_name="procedural_memory",
            embedding_function=embeddings,
            persist_directory=self.db_dir,
        )

    def _save_memory(self, store: Chroma, memory: str, config: RunnableConfig) -> str:
        username = config["configurable"].get("user_id")
        date = config["configurable"].get("thread_id")
        doc = Document(
            page_content=memory,
            id=str(uuid.uuid4()),
            metadata={"username": username, "date": date},
        )
        store.add_documents([doc])
        return memory

    def save_semantic_memory(self, memory: str, config: RunnableConfig) -> str:
        print_debug(f"Saving semantic memory: {memory}")
        return self._save_memory(self.semantic_memory, memory, config)

    def save_episodic_memory(self, memory: str, config: RunnableConfig) -> str:
        print_debug(f"Saving episodic memory: {memory}")
        return self._save_memory(self.episodic_memory, memory, config)

    def save_procedural_memory(self, memory: str, config: RunnableConfig) -> str:
        print_debug(f"Saving procedural memory: {memory}")
        return self._save_memory(self.procedural_memory, memory, config)

    def _search_memory(self, store: Chroma, query: str, config: RunnableConfig) -> List[str]:
        username = config["configurable"].get("user_id")
        docs = store.similarity_search(query, k=1, filter={"username": username})
        return [f"{d.page_content}, {d.metadata['date']}" for d in docs]

    def search_semantic_memory(self, query: str, config: RunnableConfig) -> List[str]:
        result = self._search_memory(self.semantic_memory, query, config)
        print_debug(f"Semantic memory search result: {result}")
        return result

    def search_episodic_memory(self, query: str, config: RunnableConfig) -> List[str]:
        result = self._search_memory(self.episodic_memory, query, config)
        print_debug(f"Episodic memory search result: {result}")
        return result

    def search_procedural_memory(self, query: str, config: RunnableConfig) -> List[str]:
        result = self._search_memory(self.procedural_memory, query, config)
        print_debug(f"Procedural memory search result: {result}")
        return result

    def get_full_long_term_memory(self, *args: Any, **kwargs: Any) -> List[str]:
        """Get all memories stored in Semantic, Episodic and Procedural Memory."""
        # Ignore any config args passed in
        memories: List[str] = []
        for name, store in [
            ("Semantic", self.semantic_memory),
            ("Episodic", self.episodic_memory),
            ("Procedural", self.procedural_memory),
        ]:
            memories.append(f"==== {name} Memory ====")
            results = store._collection.get(include=["documents", "metadatas"])
            if results:
                for doc, metadata in zip(
                    results.get("documents", []),
                    results.get("metadatas", []),
                ):
                    memories.append(f"{doc}, {metadata['date']}")
        return memories