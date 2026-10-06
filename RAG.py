from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import json
from qdrant_client import models
from retriever import retriever

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
QDRANT_PATH = BASE_DIR / "qdrant_data"

llm = ChatOpenAI(
        model="cohere/north-mini-code:free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )


def RAG_system(query, router_ans, metric_ans = None):
    
    router_inf = json.loads(router_ans)

    if metric_ans:
        system_promt = '''You are a financial analysis assistant for Apple, Microsoft and Amazon.

            You have two sources:
            1. Metrics: exact figures from official XBRL financial data. For any number (revenue, net income), trust this block over the text excerpts.
            2. Context: excerpts from SEC 10-K filings. Use it for explanations, reasons, risks and strategy.

            Answer using ONLY these two sources. If they don't contain enough information, say so clearly instead of guessing. When comparing companies, rely on the Company label in each excerpt and never attribute one company's text to another.

            Metrics:
            {metrics}

            Context:
            {context}'''
    else:
        system_promt = '''You are a financial analysis assistant for Apple, Microsoft and Amazon.

            Answer using ONLY the context below, which contains excerpts from SEC 10-K filings. Each excerpt is labeled with its Company, Section and Year.

            If the context doesn't contain enough information to answer, say so clearly instead of guessing. When comparing companies, rely on the Company label in each excerpt and never attribute one company's text to another.

            Context:
            {context}'''
    promt = ChatPromptTemplate.from_messages([
        ('system', system_promt),
        ('human', '{question}'),
    ]) 

    

    chain = promt | llm
    context = retriever(query, router_inf)
    if not context.strip() and not metric_ans:
        return ("There are no text reports in the database for the requested period. "
                "Text search is only available for FY2025 filings.")
    answer = chain.invoke({'context': context, 'question': query, 'metrics': metric_ans})
    return answer.content
