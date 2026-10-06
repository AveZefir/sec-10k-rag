import json
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from router import router
from RAG import RAG_system
from metric_tool import metric_tool
from dotenv import load_dotenv

class State(TypedDict):
    question: str
    router_ans: str
    rag_or_not: str
    metric_ans: str|None
    RAG_ans: str
    
graph = StateGraph(State)

def router_answer(state: State):
    return {'router_ans': router(state['question'])}

def metrics(state:State):
    return {'metric_ans':metric_tool(state['router_ans'])}

def is_other(state: State):
    router_inf = json.loads(state['router_ans'])
    qt = router_inf['question_type']
    if qt == 'other':
        return 'other'
    if qt in ('metric', 'comparison') and not router_inf['companies']:
        return 'other'
    return 'rag'

def RAG_ans(state: State):
    ans = RAG_system(state['question'], state['router_ans'], state['metric_ans'])
    return {'RAG_ans': ans}

def other_node(state: State):
    return{"RAG_ans": 'I am only accountable for the financial reports of Apple, Microsoft, and Amazon.'}

graph.add_node('router', router_answer)
graph.add_node('metric', metrics)
graph.add_node('rag', RAG_ans)
graph.add_node('other', other_node)

graph.add_edge(START, "router")
graph.add_conditional_edges('router', is_other,{'other': 'other', 'rag': 'metric'})
graph.add_edge('other', END)
graph.add_edge('metric','rag')
graph.add_edge('rag', END)

app = graph.compile()
