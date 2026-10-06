from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
def router(query):
    load_dotenv()

    llm = ChatOpenAI(
        model="cohere/north-mini-code:free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

    class MainInfo(BaseModel):
        """Analysis of a user's question about 10K documents"""
        companies: list[Literal['Apple','Microsoft','Amazon']] = Field(
            default = [],
            description =  
            "Companies mentioned in the question, only from the allowed values. "
            "Match names, tickers, products and brands: "
            "Apple = Apple, AAPL, Эппл, iPhone, Mac, iPad, App Store; "
            "Microsoft = Microsoft, MSFT, Майкрософт, Azure, Windows, Office, LinkedIn, Xbox; "
            "Amazon = Amazon, AMZN, Амазон, AWS, Prime, Alexa. "
            "Phrases like 'all three companies' mean all three. "
            "Never add a company that is not mentioned in the question. "
            "If none of these three is mentioned, including when the question is about "
            "any other company (e.g. Tesla, Google), return an empty list. Do not guess."
        )
        year: int|list[int]|None = Field(
            default= None,
            description= 
            "Every year explicitly written in the question, as integers, in any language "
            "('in 2023' -> [2023], 'за 2021 и 2023' -> [2021, 2023], 'FY2023' -> [2023]). "
            "Do not guess or infer years: relative words like 'last year' or 'this year' "
            "give null. If no year is written in the question, return null."
        )
        question_type: Literal["narrative", "metric", "comparison", "other"] = Field(
            description= 
            "Type of the question. "
            "comparison: the question covers TWO OR MORE COMPANIES (at least two of Apple, Microsoft, Amazon) "
            "and asks to compare them, or asks what they have in common or how they differ, on numbers or text. "
            "If only ONE company is mentioned, this is never comparison, even if the question uses words "
            "like 'vs', 'versus', 'compare' to contrast two different metrics of that one company — "
            "that case is metric or narrative instead. "
            "metric: asks for a specific number from financial statements "
            "(revenue, net income, total assets, EPS) for ONE company, possibly for several years. "
            "Questions like 'why' or 'how does X affect Y' are NOT metric. "
            "narrative: asks about the text of the report: risks, strategy, business description, "
            "reasons for changes, effects and impact, management discussion. "
            "other: not about the 10-K reports of Apple, Microsoft or Amazon, including "
            "questions about any other company, greetings and off-topic questions. "
            "If unsure between metric and narrative, choose narrative."
        )
        metric: Literal["revenue", "net_income"] | None = Field(
        default=None,
        description=
            "Financial metric the user asks about. "
            "'revenue': revenue, sales, top line, turnover, выручка, продажи, оборот. "
            "'net_income': net income, net profit, earnings, profit, чистая прибыль, прибыль. "
            "Only set for questions asking for a number. "
            "If no metric is named or it is not one of these two, return null. Do not guess."
        )
        
    router = llm.with_structured_output(MainInfo, method="function_calling")
    answer = router.invoke(query)
    if answer is None:
        answer = MainInfo(question_type="other")
    return answer.model_dump_json(indent=2)