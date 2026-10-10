# ============================================================
# LANGGRAPH MCP + RAG + HUMAN-IN-THE-LOOP BACKEND
# ============================================================

from __future__ import annotations

import asyncio
import json
import os
import queue
import tempfile
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

import requests
from ddgs import DDGS
from dotenv import load_dotenv

from fastmcp import Client, ClientGroup
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StdioTransport

from langchain.mcp import MCPAdapter
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, tool
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chatbot_mcp_rag_hitl.db"


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ============================================================
# THREAD-BASED RAG STORAGE
# ============================================================

_THREAD_RETRIEVERS: dict[str, Any] = {}
_THREAD_METADATA: dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if not thread_id:
        return None
    return _THREAD_RETRIEVERS.get(str(thread_id))


# ============================================================
# PDF INGESTION
# ============================================================

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """Index a PDF and associate its FAISS retriever with a thread."""

    if not file_bytes:
        raise ValueError("No PDF bytes received.")

    thread_id = str(thread_id)
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        documents = PyPDFLoader(temp_path).load()

        if not documents:
            raise ValueError("The PDF contains no readable pages.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks could be created from the PDF.")

        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        stored_filename = filename or os.path.basename(temp_path)

        _THREAD_RETRIEVERS[thread_id] = retriever
        _THREAD_METADATA[thread_id] = {
            "filename": stored_filename,
            "documents": len(documents),
            "chunks": len(chunks),
        }

        return {
            "success": True,
            "filename": stored_filename,
            "documents": len(documents),
            "chunks": len(chunks),
        }

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ============================================================
# WEB SEARCH
# ============================================================

@tool
def search_tool(query: str) -> str:
    """
    Search the web using DuckDuckGo.

    Use for current, recent, or web-based information.
    """

    try:
        results = DDGS().text(
            query,
            region="us-en",
            max_results=5,
        )

        if not results:
            return "No search results found."

        formatted = []

        for i, result in enumerate(results, start=1):
            formatted.append(
                f"{i}. {result.get('title', '')}\n"
                f"{result.get('body', '')}\n"
                f"URL: {result.get('href', '')}"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search failed: {e}"


# ============================================================
# STOCK TOOL
# ============================================================

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price using Alpha Vantage."""

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    if not api_key:
        return {
            "error": "ALPHAVANTAGE_API_KEY is not configured."
        }

    symbol = symbol.upper().strip()

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": api_key,
            },
            timeout=10,
        )

        response.raise_for_status()
        data = response.json()
        quote = data.get("Global Quote")

        if not quote:
            return {
                "error": data.get(
                    "Note",
                    data.get(
                        "Information",
                        "No stock data found.",
                    ),
                )
            }

        return {
            "symbol": quote.get("01. symbol"),
            "price": quote.get("05. price"),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent"),
            "trading_day": quote.get("07. latest trading day"),
        }

    except requests.RequestException as e:
        return {
            "error": f"Stock API request failed: {e}"
        }

    except Exception as e:
        return {
            "error": f"Unexpected stock tool error: {e}"
        }


# ============================================================
# RAG TOOL
# ============================================================

@tool
def rag_tool(
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> dict:
    """
    Search the PDF associated with the current LangGraph thread.

    thread_id is injected automatically from LangGraph config.
    """

    thread_id = (
        config
        .get("configurable", {})
        .get("thread_id")
    )

    if not thread_id:
        return {
            "error": "No thread_id was provided.",
            "query": query,
        }

    retriever = _get_retriever(str(thread_id))

    if retriever is None:
        return {
            "error": (
                "No PDF is indexed for this chat. "
                "Please upload a PDF first."
            ),
            "query": query,
        }

    try:
        documents = retriever.invoke(query)

        context = [
            document.page_content
            for document in documents
        ]

        metadata = [
            document.metadata
            for document in documents
        ]

        return {
            "query": query,
            "context": context,
            "metadata": metadata,
            "source_file": _THREAD_METADATA.get(
                str(thread_id),
                {},
            ).get("filename"),
        }

    except Exception as e:
        return {
            "error": f"RAG retrieval failed: {e}",
            "query": query,
        }


# ============================================================
# HUMAN-IN-THE-LOOP TWEET TOOL
# ============================================================

@tool
def post_tweet(tweet: str) -> str:
    """
    Ask the human to approve a drafted tweet.

    IMPORTANT:
    This is a dummy posting tool. No real tweet is published.

    The graph pauses at interrupt() until the frontend resumes it.

    approved=True:
        Dummy post succeeds.

    approved=False:
        The rejection is returned to the LLM so it can write
        another tweet and ask for approval again.
    """

    decision = interrupt(
        {
            "type": "tweet_approval",
            "tweet": tweet,
            "question": "Do you want to post this tweet?",
            "instructions": (
                "Choose Approve to perform the dummy post, "
                "or Reject to have the AI write another version."
            ),
        }
    )

    approved = False

    if isinstance(decision, dict):
        approved = bool(
            decision.get("approved", False)
        )
    elif isinstance(decision, bool):
        approved = decision

    if approved:
        return (
            "DUMMY_POST_SUCCESS: "
            "The human approved the tweet. "
            f"Dummy post completed for: {tweet}"
        )

    return (
        "TWEET_REJECTED: "
        "The human rejected this tweet. "
        "Write a new version with a meaningfully different "
        "wording and submit it to post_tweet for approval again."
    )


# ============================================================
# STATE
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]


# ============================================================
# MCP CONFIGURATION
# ============================================================

EXPENSE_URL = "https://charming-ace.fastmcp.app/mcp"

EXPENSE_CLIENT_ID = (
    "client_01M2CPAM1766J0A3KV0B9S6NFS"
)

EXPENSE_CALLBACK_PORT = 53592


# ============================================================
# BACKEND
# ============================================================

class MCPRAGHITLBackend:

    def __init__(self):

        self.chatbot = None
        self.adapter = None
        self.checkpointer = None
        self.tools = []

        self.exit_stack: AsyncExitStack | None = None
        self.initialized = False

        self.loop: asyncio.AbstractEventLoop | None = None
        self.loop_thread: threading.Thread | None = None

        self.ready_event = threading.Event()
        self.start_lock = threading.Lock()

        self.initialization_error: Exception | None = None

    # ========================================================
    # EVENT LOOP
    # ========================================================

    def start(self):

        if (
            self.loop is not None
            and self.loop.is_running()
        ):
            return

        with self.start_lock:

            if (
                self.loop is not None
                and self.loop.is_running()
            ):
                return

            self.ready_event.clear()
            self.initialization_error = None

            self.loop_thread = threading.Thread(
                target=self._run_event_loop,
                name="MCP-RAG-HITL-EventLoop",
                daemon=True,
            )

            self.loop_thread.start()

            if not self.ready_event.wait(timeout=30):
                raise TimeoutError(
                    "Backend event loop failed to start."
                )

            if self.initialization_error:
                raise self.initialization_error

    def _run_event_loop(self):

        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            self.ready_event.set()
            self.loop.run_forever()

        except Exception as e:
            self.initialization_error = e
            self.ready_event.set()

        finally:
            if self.loop is not None:
                try:
                    self.loop.close()
                except Exception:
                    pass

    def run_async(self, coro):

        self.start()

        if self.loop is None:
            raise RuntimeError(
                "Backend event loop was not created."
            )

        future = asyncio.run_coroutine_threadsafe(
            coro,
            self.loop,
        )

        return future.result()

    # ========================================================
    # INITIALIZE
    # ========================================================

    async def initialize(self):

        if self.initialized:
            return

        print(
            "\n=========================================="
        )
        print(
            "Initializing MCP + RAG + HITL backend..."
        )
        print(
            "=========================================="
        )

        try:

            self.exit_stack = AsyncExitStack()

            # ------------------------------------------------
            # Math MCP
            # ------------------------------------------------

            math_transport = StdioTransport(
                command="uv",
                args=[
                    "run",
                    "python",
                    "main.py",
                ],
                cwd=(
                    "C:\\Users\\91876"
                    "\\Desktop"
                    "\\Math-MCP-Server"
                ),
            )

            math_client = Client(
                math_transport
            )

            # ------------------------------------------------
            # Expense MCP
            # ------------------------------------------------

            expense_auth = OAuth(
                mcp_url=EXPENSE_URL,
                client_id=EXPENSE_CLIENT_ID,
                callback_port=EXPENSE_CALLBACK_PORT,
            )

            expense_client = Client(
                EXPENSE_URL,
                auth=expense_auth,
            )

            # ------------------------------------------------
            # MCP group
            # ------------------------------------------------

            mcp_group = ClientGroup(
                {
                    "arith": math_client,
                    "expense": expense_client,
                }
            )

            # ------------------------------------------------
            # MCP adapter
            # ------------------------------------------------

            self.adapter = (await self.exit_stack.enter_async_context(MCPAdapter(mcp_group)))

            print("Loading MCP tools...")

            mcp_tools = await self.adapter.list_tools()

            print(f"Loaded {len(mcp_tools)} MCP tools.")

            # All tools

            self.tools = [*mcp_tools,search_tool,get_stock_price,rag_tool,post_tweet,]

            print(f"Total tools available: {len(self.tools)}")

            # LLM

            llm_with_tools = llm.bind_tools(self.tools)

            # Chat node

            async def chat_node(state: ChatState,):

                def normalize_content(content):

                    if isinstance(content, str):
                        return (
                            content
                            if content.strip()
                            else "Tool completed successfully but returned no text.")

                    if content is None:
                        return ("Tool completed successfully but returned no text.")

                    if isinstance(content, list):

                        parts = []

                        for item in content:

                            if isinstance(item, str):
                                if item.strip():
                                    parts.append(item)

                            elif isinstance(item, dict):

                                text = item.get("text")

                                if text:
                                    parts.append(str(text))
                                else:
                                    try:
                                        parts.append(json.dumps(item,ensure_ascii=False,default=str,))
                                    except Exception:
                                        parts.append(str(item))

                            else:
                                parts.append(str(item))

                        result = "\n".join(parts).strip()

                        return (
                            result
                            if result
                            else "Tool completed successfully but returned no text.")

                    if isinstance(content, dict):

                        try:
                            return json.dumps(content,ensure_ascii=False,default=str,)
                        except Exception:
                            return str(content)

                    return str(content)

                normalized_messages = []

                for message in state["messages"]:

                    if isinstance(message, ToolMessage):

                        normalized_messages.append(
                            ToolMessage(content=normalize_content(message.content),
                                tool_call_id=message.tool_call_id,name=getattr(message,"name",None,),
                                status=getattr(message,"status","success",),))

                    else:
                        normalized_messages.append(message)

                system_message = SystemMessage(
                    content="""
You are a helpful AI assistant with access to MCP,
web search, stock data, PDF RAG, and a human-approved
dummy tweet-posting tool.

TOOL SELECTION:

- Use RAG for questions about the uploaded PDF.
- Use web search for current/recent internet information.
- Use stock tool for stock-price questions.
- Use MCP tools for math and expense operations.
- Use post_tweet when the user asks you to write/create a tweet
  and then post it.

TWEET WORKFLOW:

When the user asks to write and post a tweet:

1. Create a concise tweet based on the user's topic.
2. Call post_tweet with the drafted tweet.
3. The post_tweet tool will pause for human approval.
4. If approved, report that the dummy post succeeded.
5. If rejected, create a substantially different tweet.
6. Call post_tweet again.
7. Continue until the human approves.

HITL RULES:

- Never bypass human approval.
- Never claim a tweet was posted without the tool confirming it.
- Never expose interrupt payloads or raw tool JSON to the user.
- Human rejection means rewrite and request approval again.

RAG RULES:

- If the user asks about the uploaded document, use rag_tool.
- Do not substitute web search for a document question.
- Do not invent information absent from retrieved document context.
- If no PDF is indexed, tell the user to upload one.

GENERAL:

- Treat tool output as internal information.
- Explain results naturally.
- Do not expose raw API responses or internal tool structures.
- Current date: September 19, 2026.
"""
                )

                response = await llm_with_tools.ainvoke([system_message,*normalized_messages,])

                return {"messages": [response]}

            # Graph

            tool_node = ToolNode(self.tools)

            graph = StateGraph(ChatState)
            graph.add_node("chat_node",chat_node,)
            graph.add_node("tools",tool_node,)
            graph.add_edge(START,"chat_node",)
            graph.add_conditional_edges("chat_node",tools_condition,)
            graph.add_edge("tools","chat_node",)

            # Checkpointer

            print("Opening SQLite checkpointer...")

            self.checkpointer = (
                await self.exit_stack
                .enter_async_context(
                    AsyncSqliteSaver.from_conn_string(
                        str(DB_PATH)
                    )
                )
            )

            # Compile

            self.chatbot = graph.compile(checkpointer=self.checkpointer)

            self.initialized = True

            print("MCP + RAG + HITL backend initialized successfully!")

        except Exception:

            if self.exit_stack is not None:
                try:
                    await self.exit_stack.aclose()
                except Exception:
                    pass

            self.exit_stack = None
            self.adapter = None
            self.checkpointer = None
            self.chatbot = None
            self.tools = []
            self.initialized = False

            raise

    # STREAMING

    async def _produce_stream(self,user_input: str,config: dict,output_queue: queue.Queue,):

        try:

            await self.initialize()

            async for chunk in self.chatbot.astream(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ]
                },
                config=config,stream_mode="messages",version="v2",):

                output_queue.put(("data", chunk))

        except Exception as e:
            output_queue.put(("error", e))

        finally:
            output_queue.put(("done", None))

    def stream(self,user_input: str,thread_id: str,):

        self.start()

        output_queue = queue.Queue()

        config = {
            "configurable": {"thread_id": thread_id,},
            "metadata": {"thread_id": thread_id,},"run_name": "chat_turn",}

        future = asyncio.run_coroutine_threadsafe(self._produce_stream(user_input,config,output_queue,),self.loop,)

        try:

            while True:

                message_type, data = output_queue.get()

                if message_type == "data":
                    yield data

                elif message_type == "error":
                    raise data

                elif message_type == "done":
                    break

        finally:

            if not future.done():
                future.cancel()

    # RESUME HITL

    async def _resume(self,thread_id: str,approved: bool):

        await self.initialize()

        config = {
            "configurable": {"thread_id": thread_id,},
            "metadata": {"thread_id": thread_id,},"run_name": "tweet_approval"}

        # Import locally to keep the main import section clean.
        from langgraph.types import Command

        return await self.chatbot.ainvoke(
            Command(resume={"approved": approved,}),config=config,)

    def resume(self,thread_id: str,approved: bool,):

        return self.run_async(self._resume(thread_id,approved,))
    
    # STATE

    async def _get_state(self,thread_id: str,):

        await self.initialize()

        return await self.chatbot.aget_state({"configurable": {"thread_id": thread_id,}})

    def get_state(self,thread_id: str,):

        return self.run_async(self._get_state(thread_id))

    # INTERRUPT

    def get_interrupt(self,thread_id: str,):

        state = self.get_state(thread_id)

        for task in state.tasks:

            if task.interrupts:

                return task.interrupts[0].value

        return None

    # THREADS

    async def _get_all_threads(self):

        await self.initialize()

        threads = set()

        async for checkpoint in self.checkpointer.alist(None):

            if checkpoint is None:
                continue

            config = checkpoint.config or {}

            configurable = config.get("configurable",{})

            thread_id = configurable.get("thread_id")

            if thread_id:
                threads.add(thread_id)

        return list(threads)

    def get_all_threads(self):
        return self.run_async(self._get_all_threads())

    # SHUTDOWN

    async def _shutdown(self):

        if self.exit_stack is not None:

            try:
                await self.exit_stack.aclose()
            except Exception as e:
                print(f"Shutdown warning: {e}")

        self.exit_stack = None
        self.adapter = None
        self.checkpointer = None
        self.chatbot = None
        self.tools = []
        self.initialized = False

    def shutdown(self):

        if (self.loop is None or not self.loop.is_running()):
            return

        try:

            future = asyncio.run_coroutine_threadsafe(self._shutdown(),self.loop,)

            future.result(timeout=10)

        except Exception as e:

            print(f"Shutdown error: {e}")

        finally:

            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception:
                pass

# GLOBAL BACKEND

backend = MCPRAGHITLBackend()

# PUBLIC HELPERS

def ingest_document(file_bytes: bytes,thread_id: str,filename: Optional[str] = None,):
    return ingest_pdf(file_bytes=file_bytes,thread_id=thread_id,filename=filename,)

def thread_has_document(thread_id: str,) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str,) -> dict:
    return _THREAD_METADATA.get(str(thread_id),{},)


def retrieve_all_threads():
    return backend.get_all_threads()

def get_conversation_state(thread_id: str):
    return backend.get_state(thread_id)

def stream_chat(user_input: str,thread_id: str):
    yield from backend.stream(user_input,thread_id,)

def resume_chat(thread_id: str,approved: bool):
    return backend.resume(thread_id,approved,)


def get_pending_interrupt(thread_id: str):
    return backend.get_interrupt(thread_id)
