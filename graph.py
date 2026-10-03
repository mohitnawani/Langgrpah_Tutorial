"""
Supervisor + 4 worker agents (LangGraph).

Install:
    pip install langgraph langchain langchain-openai langchain-core pydantic
    export OPENAI_API_KEY=...

Flow:
    user -> SUPERVISOR -> (action | database | rag | utility) -> back to SUPERVISOR
                       -> ... repeat until FINISH -> RESPOND -> final answer
"""

import sqlite3
from datetime import date
from typing import Literal
from dotenv import load_dotenv
import sqlite3


from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.agent create_react_agent

from langchain_google_genai import ChatGoogleGenerativeAI

# Swap the model here (e.g. "anthropic:claude-sonnet-4-5", "google_genai:gemini-2.0-flash")


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# ─────────────────────────────────────────────────────────────
# 1) ACTION AGENT  – does things (mocked side effects)
# ─────────────────────────────────────────────────────────────
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to someone."""
    print(f"   📧 EMAIL -> {to} | {subject}")
    return f"Email sent to {to} with subject '{subject}'."


@tool
def create_calendar_event(title: str, when: str) -> str:
    """Create a calendar event. `when` is free text like '2026-10-05 15:00'."""
    print(f"   📅 EVENT -> {title} @ {when}")
    return f"Event '{title}' created for {when}."


@tool
def create_task(task: str, assignee: str) -> str:
    """Create a to-do task and assign it to someone."""
    print(f"   ✅ TASK -> {task} (for {assignee})")
    return f"Task '{task}' assigned to {assignee}."


action_agent = create_react_agent(
    llm,
    tools=[send_email, create_calendar_event, create_task],
    prompt="You perform actions (email, calendar, tasks). Do exactly what's asked, then report briefly.",
)


# ─────────────────────────────────────────────────────────────
# 2) DATABASE AGENT – talks to SQLite
# ─────────────────────────────────────────────────────────────
db = sqlite3.connect(":memory:", check_same_thread=False)
db.executescript("""
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary INTEGER, email TEXT);
INSERT INTO employees (name, dept, salary, email) VALUES
 ('Asha',  'Engineering', 120000, 'asha@corp.com'),
 ('Ravi',  'Engineering', 100000, 'ravi@corp.com'),
 ('Meera', 'HR',           80000, 'meera@corp.com'),
 ('Karan', 'Sales',        90000, 'karan@corp.com');
""")


@tool
def get_schema() -> str:
    """Return the database schema. Call this before writing SQL."""
    rows = db.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    return "\n".join(r[0] for r in rows)


@tool
def run_select_query(sql: str) -> str:
    """Run a READ-ONLY SQL SELECT query and return rows."""
    if not sql.strip().lower().startswith("select"):
        return "Error: only SELECT queries are allowed."
    try:
        return str(db.execute(sql).fetchall())
    except Exception as e:
        return f"SQL error: {e}"


@tool
def add_employee(name: str, dept: str, salary: int, email: str) -> str:
    """Insert a new employee row."""
    db.execute(
        "INSERT INTO employees (name, dept, salary, email) VALUES (?,?,?,?)",
        (name, dept, salary, email),
    )
    db.commit()
    return f"Added {name} to {dept}."


database_agent = create_react_agent(
    llm,
    tools=[get_schema, run_select_query, add_employee],
    prompt="You are a database agent. Check the schema first, then write SQL. Never guess data.",
)


# ─────────────────────────────────────────────────────────────
# 3) RAG AGENT – answers from company documents
# ─────────────────────────────────────────────────────────────
DOCS = [
    Document(
        page_content="Leave policy: Employees get 24 paid leaves per year. Unused leaves carry over up to 10 days.",
        metadata={"title": "Leave Policy"},
    ),
    Document(
        page_content="Work from home: Employees may work remotely 2 days per week with manager approval.",
        metadata={"title": "WFH Policy"},
    ),
    Document(
        page_content="Expenses: Submit reimbursement within 30 days. Claims above 5000 INR need manager approval.",
        metadata={"title": "Expense Policy"},
    ),
]
vector_store = InMemoryVectorStore(OpenAIEmbeddings())
vector_store.add_documents(DOCS)


@tool
def search_company_docs(query: str) -> str:
    """Semantic search over company policy documents. Returns the top matching passages."""
    hits = vector_store.similarity_search(query, k=2)
    return "\n".join(f"[{d.metadata['title']}] {d.page_content}" for d in hits)


@tool
def list_document_titles() -> str:
    """List all available policy document titles."""
    return ", ".join(d.metadata["title"] for d in DOCS)


rag_agent = create_react_agent(
    llm,
    tools=[search_company_docs, list_document_titles],
    prompt="Answer ONLY from the retrieved documents. If not found, say you don't know.",
)


# ─────────────────────────────────────────────────────────────
# 4) UTILITY AGENT – math + date
# ─────────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression like '120000 * 0.1'."""
    if not set(expression) <= set("0123456789+-*/(). %"):
        return "Error: invalid characters."
    return str(eval(expression, {"__builtins__": {}}))


@tool
def get_today() -> str:
    """Get today's date (YYYY-MM-DD)."""
    return date.today().isoformat()


utility_agent = create_react_agent(
    llm,
    tools=[calculator, get_today],
    prompt="You do calculations and date lookups. Always use your tools for math.",
)


# ─────────────────────────────────────────────────────────────
# SUPERVISOR – decides who works next
# ─────────────────────────────────────────────────────────────
WORKERS = {
    "action_agent": action_agent,  # email, calendar, tasks
    "database_agent": database_agent,  # SQL on employees
    "rag_agent": rag_agent,  # company policy docs
    "utility_agent": utility_agent,  # math, date
}


class State(MessagesState):
    next: str  # supervisor's routing decision, stored in the graph state


class Route(BaseModel):
    """Supervisor's structured decision."""

    next: Literal[
        "action_agent", "database_agent", "rag_agent", "utility_agent", "FINISH"
    ]
    reason: str


SUPERVISOR_PROMPT = """You are a supervisor managing these workers:
- action_agent: send emails, create calendar events, create tasks
- database_agent: query/insert employee data (name, dept, salary, email)
- rag_agent: answer questions about company policies (leave, WFH, expenses)
- utility_agent: math and today's date

Look at the conversation so far. Pick the ONE worker who should act next.
If the user's request is fully done, choose FINISH.
Break multi-step requests into steps (e.g. fetch data first, then act on it)."""

supervisor_llm = llm.with_structured_output(Route)


def supervisor_node(state: State):
    decision = supervisor_llm.invoke(
        [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    )
    print(f"🧭 Supervisor -> {decision.next}  ({decision.reason})")
    return {"next": decision.next}


def make_worker_node(name: str, agent):
    def node(state: State):
        result = agent.invoke({"messages": state["messages"]})
        answer = result["messages"][-1].content
        print(f"   🤖 {name}: {answer}")
        # Report back as a named message so the supervisor can read it
        return {"messages": [HumanMessage(content=answer, name=name)]}

    return node


def respond_node(state: State):
    final = llm.invoke(
        [
            SystemMessage(
                content="Write a short final answer to the user's original request "
                "using the work done by the workers above."
            )
        ]
        + state["messages"]
    )
    return {"messages": [final]}


# ─────────────────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────
graph = StateGraph(State)
graph.add_node("supervisor", supervisor_node)
graph.add_node("respond", respond_node)
for name, agent in WORKERS.items():
    graph.add_node(name, make_worker_node(name, agent))
    graph.add_edge(name, "supervisor")  # every worker reports back to supervisor

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    lambda s: s["next"],  # read the routing decision
    {**{n: n for n in WORKERS}, "FINISH": "respond"},
)
graph.add_edge("respond", END)

app = graph.compile()


def ask(question: str):
    print(f"\n{'=' * 60}\n❓ {question}\n{'=' * 60}")
    out = app.invoke(
        {"messages": [HumanMessage(content=question)]}, {"recursion_limit": 15}
    )
    print(f"\n💬 FINAL: {out['messages'][-1].content}")


if __name__ == "__main__":
    ask("How many leaves do employees get per year?")  # rag only
    ask(
        "Who earns the most in Engineering, and what is 10% of their salary?"
    )  # db -> utility
    ask(
        "Find Meera's email from the database and email her the WFH policy summary."
    )  # db -> rag -> action
