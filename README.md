# 🤖 Multi-Utility AI Agent — MCP, RAG & Human-in-the-Loop

An interactive, agentic AI assistant built using **LangGraph, Groq, Model Context Protocol (MCP), Retrieval-Augmented Generation (RAG), and Streamlit**.

The application combines LLM-powered reasoning, external tool integration, document-based question answering, and human approval workflows in a unified conversational interface.

Unlike a basic chatbot, this application uses an agentic workflow to select tools, retrieve relevant document context, interact with external services, and pause for human approval when required.

---

## ✨ Key Features

### 🧠 1. Agentic AI with LangGraph
- Built an agentic conversational workflow using LangGraph.
- Integrates LLM-based responses with tool-calling capabilities.
- Routes requests through an agent-and-tools execution loop.
- Supports multi-step interactions involving external tools.

### 📄 2. Retrieval-Augmented Generation (RAG)
- Upload PDF documents through the Streamlit interface.
- Extract and process document text using PyPDFLoader.
- Split documents into manageable chunks using RecursiveCharacterTextSplitter.
- Generate semantic embeddings using Hugging Face sentence-transformer models.
- Store and retrieve document vectors using FAISS.
- Retrieve relevant document chunks to provide context-grounded responses.

**RAG Pipeline:**

```text
PDF Upload
    |
    v
Document Text Extraction
    |
    v
Text Chunking
    |
    v
Hugging Face Embeddings
    |
    v
FAISS Vector Store
    |
    v
Similarity-Based Retrieval
    |
    v
LLM + Retrieved Context
    |
    v
Context-Grounded Answer
```

### 🔌 3. Model Context Protocol (MCP)
- Integrates MCP tools into the agent workflow.
- Uses FastMCP for MCP server and client functionality.
- Supports interaction with external tools and services.
- Demonstrates integration of multiple utilities through a common agent interface.

### 🔍 4. Web Search
- Integrates DuckDuckGo search functionality.
- Enables the agent to retrieve information from web search results.
- Supports information-seeking workflows beyond uploaded documents.

### 📈 5. Stock Information Tool
- Integrates a stock-information tool using an external API.
- Allows the agent to retrieve stock-related information through tool calls.
- Demonstrates API-based external data integration.

### 👤 6. Human-in-the-Loop (HITL)
- Implements a human approval workflow using LangGraph interrupts.
- Pauses the workflow when a tweet draft requires approval.
- Allows the user to approve or reject the proposed action through the Streamlit interface.
- Demonstrates human oversight before proceeding with an action.

**Approval Workflow:**

```text
User Request
     |
     v
LangGraph Agent
     |
     v
Generate Tweet Draft
     |
     v
Human Approval Required
     |
     +------------------+
     |                  |
     v                  v
  Approve             Reject
     |                  |
     v                  v
Continue Workflow   Stop/Reject Action
```

> **Note:** The current tweet-posting tool is a demonstration tool. It does not publish tweets to a real social media account.

### 💾 7. Persistent Conversation Checkpointing
- Uses SQLite-based checkpointing with LangGraph.
- Maintains workflow state across execution steps.
- Supports conversation recovery and resuming interrupted workflows.
- Enables human-in-the-loop interactions to continue from saved checkpoints.

### 💬 8. Interactive Streamlit Interface
- Provides a conversational chatbot interface.
- Supports PDF uploads and document indexing.
- Displays streaming AI responses.
- Supports conversation and thread selection.
- Provides approval and rejection controls for tweet drafts.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────────┐
                    │    Streamlit Frontend   │
                    │                         │
                    │ Chat | PDF Upload       │
                    │ Threads | HITL Approval │
                    └────────────┬────────────┘
                                 |
                                 v
                    ┌─────────────────────────┐
                    │     LangGraph Agent     │
                    │                         │
                    │ Agent Node              │
                    │ Tool Execution          │
                    │ Workflow Routing        │
                    │ Interrupts & Checkpoints│
                    └────────────┬────────────┘
                                 |
                 ┌───────────────┼───────────────┐
                 |               |               |
                 v               v               v
        ┌────────────────┐ ┌─────────────┐ ┌──────────────┐
        │   Groq LLM     │ │  RAG Tools  │ │   MCP Tools  │
        │                │ │             │ │              │
        │ Reasoning      │ │ PDF Loading │ │ FastMCP      │
        │ Tool Calling   │ │ Embeddings  │ │ External APIs│
        │ Responses      │ │ FAISS Search│ │ Utilities    │
        └────────────────┘ └─────────────┘ └──────────────┘
                                 |
                                 v
                    ┌─────────────────────────┐
                    │  SQLite Checkpointing   │
                    │                         │
                    │ Conversation State      │
                    │ Workflow Recovery       │
                    └─────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| LLM Provider | Groq |
| LLM Integration | LangChain |
| Agent Framework | LangGraph |
| MCP Integration | FastMCP |
| RAG Framework | LangChain |
| Document Processing | PyPDFLoader |
| Text Chunking | RecursiveCharacterTextSplitter |
| Embeddings | Hugging Face Sentence Transformers |
| Vector Search | FAISS |
| Frontend | Streamlit |
| Conversation Checkpointing | SQLite |
| Web Search | DuckDuckGo |
| External Data Integration | Stock API |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

The project is organized around a Streamlit frontend, a LangGraph agent backend, and MCP server integrations.

A simplified representation is:

```text
multi-utility-ai-agent/
│
├── streamlit_chatbot_hitl.py
│   └── Streamlit interface
│
├── <langgraph_backend_file>.py
│   └── Agent workflow, tools, RAG, MCP integration,
│       human-in-the-loop and checkpointing
│
├── <mcp_server_files>/
│   └── MCP server implementations
│
├── pyproject.toml
│   └── Project dependencies and configuration
│
├── .env
│   └── Local environment variables (not committed)
│
├── .gitignore
│
└── README.md
```

Replace the placeholder filenames and folders with the actual names in your repository.

---

## ⚙️ Prerequisites

Before running the project, make sure you have:

- Python installed.
- Git installed.
- `uv` installed if you want to use the project's `uv` workflow.
- A Groq API key.
- Access to any external APIs required by the enabled tools.
- The dependencies specified in the project configuration.

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Navigate into the project directory:

```bash
cd YOUR_REPOSITORY
```

### 2. Install dependencies

If the project uses `uv` and contains a valid `pyproject.toml`:

```bash
uv sync
```

Alternatively, install the dependencies using the package manager and dependency configuration used by your project.

### 3. Configure environment variables

Create a `.env` file in the project root.

Example:

```env
GROQ_API_KEY=your_groq_api_key

# Add other API credentials required by your enabled tools.
# Use the exact variable names expected by your code.
```

If your stock tool or remote MCP service requires additional credentials, configure them according to its implementation.

**Security:** Never commit your `.env` file, API keys, OAuth tokens, or other secrets to GitHub.

### 4. Configure MCP servers

Review the MCP client configuration and ensure that:

- Local MCP server paths match your machine.
- Required server dependencies are installed.
- Remote MCP endpoints are reachable.
- OAuth credentials are configured if required.

If your MCP server uses a local Windows path, replace the machine-specific path with a portable configuration before sharing the project.

### 5. Run the application

If your Streamlit entry point is `streamlit_chatbot_hitl.py`, run:

```bash
uv run streamlit run streamlit_chatbot_hitl.py
```

Alternatively:

```bash
streamlit run streamlit_chatbot_hitl.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

---

## 📚 How to Use

### Step 1: Start the application
Launch the Streamlit interface.

### Step 2: Upload a PDF
Upload a document and run the document indexing workflow.

### Step 3: Ask questions
Ask questions about the uploaded document and receive responses grounded in retrieved context.

### Step 4: Use agent tools
Submit requests that can use the configured web search, stock-information, or MCP tools.

### Step 5: Review human approval requests
When the workflow produces a tweet draft requiring approval, review the draft and choose the available approval or rejection action.

### Step 6: Resume conversations
Use the conversation interface to continue supported threads and resume workflows using their saved checkpoint state.

---

## 🧠 Core Concepts Demonstrated

This project brings together several important concepts in modern AI application development:

- **Agentic AI:** LLM-driven tool selection and execution.
- **Stateful workflows:** Graph-based execution and conditional routing.
- **RAG:** Grounding responses in retrieved document context.
- **Vector search:** Semantic retrieval using embeddings and FAISS.
- **MCP:** Standardized integration with external tools.
- **Human-in-the-Loop:** Human approval before proceeding with selected actions.
- **Persistent checkpointing:** Saving workflow state using SQLite.
- **API integration:** Connecting AI workflows to external services.
- **Interactive applications:** Building a conversational UI with Streamlit.

---

## 🔐 Security Considerations

For safe development and deployment:

- Store secrets in environment variables or a dedicated secret manager.
- Never commit API keys or OAuth tokens.
- Validate uploaded files before processing them.
- Restrict access to sensitive documents and conversation data.
- Apply appropriate authentication and authorization before exposing the application publicly.
- Review tool permissions before enabling external actions.
- Keep human approval enabled for sensitive operations.
- Use a portable configuration rather than hard-coded local file paths.

The current project is a development application. Additional security controls should be implemented before exposing it to untrusted users.

---

## 🚧 Current Scope & Limitations

- PDF retrieval depends on the uploaded documents and the quality of extracted text.
- Responses may still contain incorrect information and should be verified when accuracy is important.
- External tools depend on their APIs, network connectivity, and credentials.
- MCP servers must be configured and reachable.
- The current tweet-posting tool is a demonstration implementation and does not publish real tweets.
- SQLite checkpointing provides local persistence; it is not a distributed production database.
- Production deployment requires additional configuration, testing, access control, and operational monitoring.

---

## 🔮 Future Improvements

Potential extensions include:

- [ ] Add a FastAPI backend to expose agent functionality through REST endpoints.
- [ ] Containerize the application using Docker.
- [ ] Add automated testing with Pytest.
- [ ] Add CI/CD workflows for automated testing and deployment.
- [ ] Deploy the application to a cloud platform.
- [ ] Add authentication and user-specific conversation isolation.
- [ ] Add structured logging and error monitoring.
- [ ] Improve retrieval using metadata filtering and reranking.
- [ ] Add evaluation datasets for measuring RAG quality.
- [ ] Add production-ready database and checkpoint storage.
- [ ] Integrate a real social media API with appropriate permissions and approval controls.

These are planned enhancements, not features currently claimed as implemented.

---

## 🎯 Project Objective

The goal of this project is to explore how LLMs, agent orchestration, document retrieval, external tool integration, and human oversight can work together to create a more capable AI assistant.

It demonstrates the development of a tool-enabled, stateful AI application beyond a traditional question-answering chatbot.

---

## 👨‍💻 Author

**Sachin Singh**

- GitHub: https://github.com/Sachin-Singh-05
- LinkedIn: https://www.linkedin.com/in/sachin-singh-477a44322

---

⭐ If you find this project useful, consider giving the repository a star!
