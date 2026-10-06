from pathlib import Path
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent
QDRANT_PATH = BASE_DIR / "qdrant_data"

client = QdrantClient(path=str(QDRANT_PATH))
embed_model = HuggingFaceEmbeddings(
    model_name="Supabase/bge-small-en",
    encode_kwargs={"normalize_embeddings": True},
    query_encode_kwargs={
        "normalize_embeddings": True,
        "prompt": "Represent this sentence for searching relevant passages: ",
    },
)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="10k_collection",
    embedding=embed_model,
)


def retriever(query, router_inf):
    companies = router_inf['companies']
    year = router_inf['year']

    if year is None:
        years = []
    elif isinstance(year, list):
        years = [str(y) for y in year]
    else:
        years = [str(year)]

    all_docs = []
    for c in companies or [None]:
        conditions = []
        if c:
            conditions.append(
                models.FieldCondition(key='metadata.company', match=models.MatchAny(any=[c]))
            )
        if years:
            conditions.append(
                models.FieldCondition(key='metadata.year', match=models.MatchAny(any=years))
            )
        # эти три строки ВНУТРИ цикла
        fltr = models.Filter(must=conditions) if conditions else None
        k = 3 if c else 5
        all_docs += vector_store.similarity_search(query, k=k, filter=fltr)

    context = ''
    for doc in all_docs:
        context += (
            f"\n\nCompany: {doc.metadata['company']}| "
            f"Section: {doc.metadata['section']}| "
            f"Year: {doc.metadata['year']}\n{doc.page_content}\n---"
        )
    return context