# ============================================================
# STREAMLIT MCP + RAG + HUMAN-IN-THE-LOOP CHATBOT
# ============================================================

import uuid

import streamlit as st

from langchain_core.messages import (AIMessage,AIMessageChunk,HumanMessage,ToolMessage,)

from langgraph_backend_hitl import (
    retrieve_all_threads,
    get_conversation_state,
    stream_chat,
    ingest_document,
    thread_document_metadata,
    resume_chat,
    get_pending_interrupt,
)

# PAGE CONFIG

st.set_page_config(
    page_title="MCP + RAG + HITL Agent",
    page_icon="🤖",
    layout="wide",
)

# UTILITY

def generate_thread_id():
    return str(uuid.uuid4())


def extract_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                text = item.get("text")

                if text:
                    parts.append(str(text))

        return "".join(parts)

    return ""

# THREAD MANAGEMENT

def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(
            thread_id
        )


def reset_chat():

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


def load_conversation(thread_id):

    state = get_conversation_state(thread_id)

    if state is None:
        return []

    messages = state.values.get("messages",[],)

    conversation = []

    for message in messages:

        if isinstance(message, ToolMessage):
            continue

        if isinstance(message, HumanMessage):

            text = extract_text(message.content)

            if text:
                conversation.append(
                    {
                        "role": "user",
                        "content": text,
                    }
                )

        elif isinstance(message, AIMessage):

            text = extract_text(message.content)

            if text:
                conversation.append(
                    {
                        "role": "assistant",
                        "content": text,
                    }
                )

    return conversation

# SESSION STATE

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:

    try:
        st.session_state["chat_threads"] = (retrieve_all_threads())

    except Exception as e:

        st.session_state["chat_threads"] = []

        st.sidebar.error(f"Could not load conversations: {e}")

add_thread(st.session_state["thread_id"])

thread_id = st.session_state["thread_id"]

# SIDEBAR

st.sidebar.title("🤖 MCP + RAG + HITL")

st.sidebar.caption("Groq • LangGraph • MCP • RAG • Human Approval")

st.sidebar.markdown(f"**Thread:** `{thread_id}`")

if st.sidebar.button("New Chat",use_container_width=True,):

    reset_chat()
    st.rerun()

# DOCUMENT UPLOAD

st.sidebar.header(
    "📄 PDF RAG"
)

doc_meta = thread_document_metadata(thread_id)

if doc_meta:

    st.sidebar.success(
        f"📚 {doc_meta.get('filename', 'PDF')}\n\n"
        f"Pages: {doc_meta.get('documents', 0)}\n\n"
        f"Chunks: {doc_meta.get('chunks', 0)}"
    )

else:

    st.sidebar.info("No PDF indexed for this chat.")


uploaded_pdf = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"],
    key=f"pdf_{thread_id}",
)


if uploaded_pdf is not None:

    current_filename = doc_meta.get("filename")

    if current_filename == uploaded_pdf.name:

        st.sidebar.info(f"`{uploaded_pdf.name}` is already indexed.")

    else:

        with st.sidebar.status(
            "📚 Indexing PDF...",
            expanded=True,
        ) as status:

            try:

                summary = ingest_document(
                    file_bytes=uploaded_pdf.getvalue(),
                    thread_id=thread_id,
                    filename=uploaded_pdf.name,
                )

                status.update(
                    label="✅ PDF indexed",
                    state="complete",
                    expanded=False,
                )

                st.sidebar.success(
                    f"Indexed `{summary['filename']}`\n\n"
                    f"Pages: {summary['documents']}\n\n"
                    f"Chunks: {summary['chunks']}"
                )

                st.rerun()

            except Exception as e:

                status.update(
                    label="❌ PDF indexing failed",
                    state="error",
                    expanded=True,
                )

                st.sidebar.error(
                    str(e)
                )

# CONVERSATIONS

st.sidebar.header("💬 Conversations")

threads = list(reversed(st.session_state["chat_threads"]))

if not threads:
    st.sidebar.caption("No previous conversations.")

else:

    for index, chat_thread in enumerate(threads,start=1,):

        if (chat_thread== st.session_state["thread_id"]):

            label = f"🟢 Chat {index}"

        else:

            label = f"Chat {index}"

        if st.sidebar.button(label,key=f"thread_{chat_thread}",use_container_width=True,):

            try:

                st.session_state["thread_id"] = (chat_thread)

                st.session_state["message_history"] = load_conversation(chat_thread)

                st.rerun()

            except Exception as e:

                st.sidebar.error(f"Could not load chat: {e}")

# MAIN HEADER

st.title("🤖 Multi-Utility Agent")

st.caption("MCP + PDF RAG + Human-in-the-Loop Tweet Posting")


# ACTIVE DOCUMENT

active_doc = thread_document_metadata(thread_id)

if active_doc:

    st.info(
        f"📄 **Active PDF:** "
        f"{active_doc.get('filename', 'Unknown')} "
        f"• {active_doc.get('documents', 0)} pages "
        f"• {active_doc.get('chunks', 0)} chunks"
    )

# CHAT HISTORY

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

# HUMAN APPROVAL CHECK

pending_interrupt = get_pending_interrupt(thread_id)

if (
    isinstance(pending_interrupt, dict)
    and pending_interrupt.get("type")== "tweet_approval"):

    tweet = pending_interrupt.get("tweet","",)

    st.divider()

    st.subheader("🧑 Human Approval Required")

    st.warning(
        "The AI wants to post this tweet. "
        "No real post will be made — this is a dummy tool."
        )

    st.markdown("### 🐦 Proposed Tweet")

    st.info(tweet)

    st.write(
        pending_interrupt.get("question","Do you want to post this tweet?",))

    col1, col2 = st.columns(2)

    with col1:

        approve = st.button(
            "✅ Yes — Post",
            type="primary",
            use_container_width=True,
            key="approve_tweet",
        )

    with col2:

        reject = st.button(
            "❌ No — Rewrite",
            use_container_width=True,
            key="reject_tweet",
        )

    if approve:

        try:

            with st.spinner(
                "Resuming agent and performing dummy post..."
            ):

                result = resume_chat(thread_id,approved=True,)

            st.success(
                "✅ Human approved the tweet. "
                "Dummy post completed."
            )

            st.rerun()

        except Exception as e:

            st.error(f"Could not resume agent: {e}")

    if reject:

        try:

            with st.spinner(
                "Human rejected the tweet. AI is writing a new version..."
            ):

                result = resume_chat(thread_id,approved=False,)

            st.rerun()

        except Exception as e:

            st.error(f"Could not resume agent: {e}")

# CHAT INPUT

if not (
    isinstance(pending_interrupt,dict,)
    and pending_interrupt.get("type")== "tweet_approval"):

    user_input = st.chat_input("Ask about your PDF, use MCP tools, or ask me to write a tweet...")

    if user_input:

        st.session_state["message_history"].append({
                "role": "user",
                "content": user_input,
            })

        with st.chat_message("user"):

            st.markdown(user_input)

        with st.chat_message("assistant"):

            response_placeholder = st.empty()

            tool_status = None
            assistant_text = ""

            try:

                for chunk in stream_chat(user_input,thread_id,):

                    if not isinstance(chunk,dict,):
                        continue

                    if chunk.get("type") == "error":

                        raise RuntimeError(
                            str(
                                chunk.get("error","Unknown agent error.",)))

                    if chunk.get("type") != "messages":
                        continue

                    data = chunk.get("data")

                    if not data:
                        continue

                    if not isinstance(data,(tuple, list),):
                        continue

                    if len(data) < 1:
                        continue

                    message_chunk = data[0]

                    # Tool execution

                    if isinstance(message_chunk,ToolMessage,):

                        tool_name = (getattr(message_chunk,"name",None,)or "tool")

                        if tool_status is None:

                            tool_status = st.status(f"🔧 Using `{tool_name}`...",expanded=True,)

                        else:

                            tool_status.update(label=(f"🔧 Using `{tool_name}`..."),state="running",expanded=True,)

                        continue

                    # AI response

                    if isinstance(message_chunk,(AIMessage,AIMessageChunk,),):

                        text = extract_text(message_chunk.content)

                        if not text:
                            continue

                        assistant_text += text

                        response_placeholder.markdown(assistant_text)

                if tool_status is not None:

                    tool_status.update(
                        label="✅ Tool finished",
                        state="complete",
                        expanded=False,
                    )

                # ------------------------------------------------
                # IMPORTANT:
                # The graph may have stopped at interrupt()
                # after the tweet tool was called.
                # ------------------------------------------------

                pending_after_stream = (get_pending_interrupt(thread_id))

                if (
                    isinstance(pending_after_stream,dict)and pending_after_stream.get("type")== "tweet_approval"):

                    # The approval card will be rendered
                    # on the next Streamlit rerun.
                    st.rerun()

                if not assistant_text:

                    assistant_text = ("The agent completed the request.")

                    response_placeholder.markdown(assistant_text)

                st.session_state[
                    "message_history"
                ].append({
                        "role": "assistant",
                        "content": assistant_text,
                    })

            except Exception as e:

                if tool_status is not None:

                    tool_status.update(
                        label="❌ Agent error",
                        state="error",
                        expanded=True,
                    )

                error_text = (
                    f"❌ Agent error: {e}"
                )

                response_placeholder.error(error_text)

                st.session_state[
                    "message_history"
                ].append({
                        "role": "assistant",
                        "content": error_text,
                    })

# FOOTER

st.divider()

st.caption(
    "🔐 Tweet posting always requires human approval. "
    "The current post tool is a dummy action."
)
