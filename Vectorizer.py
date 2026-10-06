#chuncker&embeddings
import os
import json
from langchain_core.documents import Document
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
path10k = DATA_DIR / "all_10k_clean.json"
with open(path10k, "r", encoding="utf-8") as f:
    all_10k = json.load(f)
client = QdrantClient(path=str(BASE_DIR / "qdrant_data"))

def json2doc(file):
    return [Document(page_content=item["text"], metadata={"company": item["company"], "section": item["section"], "year": item["year"]}) for item in file]

embed_model = HuggingFaceEmbeddings(model_name="Supabase/bge-small-en",
                                    encode_kwargs={"normalize_embeddings": True},
                                    query_encode_kwargs = {"normalize_embeddings": True, "prompt": "Represent this sentence for searching relevant passages: "})
collection_is_new = not client.collection_exists("10k_collection")
if collection_is_new:
    client.create_collection(
        collection_name="10k_collection",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="10k_collection",
    embedding=embed_model,
)

if collection_is_new:
    splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 50)
    all_10k_doc = json2doc(all_10k)
    chunks = splitter.split_documents(all_10k_doc)
    print(len(chunks))
    vector_store.add_documents(chunks)
    
