from pathlib import Path
from typing import Annotated, TypedDict
import json
import os
import sqlite3
import uuid
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

load_dotenv(Path(__file__).with_name(".env"))

# -------------------
# 1. LLM
# -------------------
google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    google_api_key=google_api_key,
)

# -------------------
# 2. Tools
# -------------------
# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}




@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()



tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


# -------------------
# 8. Uvicorn ASGI app
# -------------------
async def _read_json_body(receive):
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break

    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


async def _send_json(send, status_code, data):
    content = json.dumps(data).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(content)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": content})


async def app(scope, receive, send):
    if scope["type"] != "http":
        await _send_json(send, 400, {"error": "Only HTTP requests are supported"})
        return

    method = scope["method"]
    path = scope["path"]

    if method == "GET" and path == "/":
        await _send_json(send, 200, {"service": "langgraph-chatbot", "status": "ok"})
        return

    if method == "GET" and path == "/health":
        await _send_json(send, 200, {"status": "ok"})
        return

    if method == "GET" and path == "/threads":
        await _send_json(send, 200, {"threads": retrieve_all_threads()})
        return

    if method == "POST" and path == "/chat":
        try:
            payload = await _read_json_body(receive)
            user_message = payload.get("message")
            if not user_message:
                await _send_json(send, 400, {"error": "Field 'message' is required"})
                return

            thread_id = payload.get("thread_id") or str(uuid.uuid4())
            result = chatbot.invoke(
                {"messages": [HumanMessage(content=user_message)]},
                config={"configurable": {"thread_id": thread_id}},
            )
            await _send_json(
                send,
                200,
                {
                    "thread_id": thread_id,
                    "response": result["messages"][-1].content,
                },
            )
        except Exception as exc:
            await _send_json(send, 500, {"error": str(exc)})
        return

    await _send_json(send, 404, {"error": "Not found"})
