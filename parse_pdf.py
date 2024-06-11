import textwrap
from typing import List

from FlagEmbedding import BGEM3FlagModel
from llama_index.core import StorageContext, Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction


class ExampleEmbeddingFunction(BaseSparseEmbeddingFunction):
    def __init__(self):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    def encode_queries(self, queries: List[str]):
        outputs = self.model.encode(
            queries,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def encode_documents(self, documents: List[str]):
        outputs = self.model.encode(
            documents,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def _to_standard_dict(self, raw_output):
        result = {}
        for k in raw_output:
            result[int(k)] = raw_output[k]
        return result


# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

print("Document ID:", documents[0].doc_id)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
Settings.llm = Ollama(base_url="http://10.8.0.100:11434", model="llama3", request_timeout=360.0)

# Create an index over the documnts
vector_store = MilvusVectorStore(
    url="http://10.8.0.100:19530/",
    dim=1536,
    overwrite=True,
    enable_sparse=True,
    sparse_embedding_function=ExampleEmbeddingFunction(),
    hybrid_ranker="RRFRanker",
    hybrid_ranker_params={"k": 60},
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What is the number?")
print(textwrap.fill(str(response), 100))

response = query_engine.query("Who is the author?")
print(textwrap.fill(str(response), 100))
