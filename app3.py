import streamlit as st
from langchain-community.tools import WikipediaQueryRun
from langchain-community.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain-community.llms import Ollama
import os
import warnings
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)


# ---------------------------- PAGE CONFIG ----------------------------
st.set_page_config(page_title="Smart AI Assistant", page_icon="💡", layout="centered")

# ---------------------------- API KEY ----------------------------
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG Chatbot"
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# ---------------------------- CUSTOM CSS ----------------------------
st.markdown("""
    <style>
        .stApp {
            background-color: #fefefe;
            font-family: 'Segoe UI', sans-serif;
        }
        .chat-message {
            padding: 0.75rem;
            border-radius: 10px;
            margin: 10px 0;
            width: fit-content;
            max-width: 85%;
        }
        .user {
            background-color: #d9f4ff;
            margin-left: auto;
        }
        .assistant {
            background-color: #f3e5f5;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- HEADER ----------------------------
st.markdown("<h1 style='text-align: center;'>💡 JOHN AI SOLUTIONS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>WELCOME TO THE AI WORLD</p>", unsafe_allow_html=True)

# ---------------------------- SIDEBAR ----------------------------
st.sidebar.image("pic.png", width=150)
st.sidebar.title("🗂️Topics Discussed")

# ---------------------------- TOPIC MANAGEMENT ----------------------------
if "topic_summary" not in st.session_state:
    st.session_state["topic_summary"] = ""
if "last_topic_question" not in st.session_state:
    st.session_state["last_topic_question"] = ""
if "previous_topics" not in st.session_state:
    st.session_state["previous_topics"] = []

def summarize_topic(question: str):
    topic_prompt = f"Summarize the following question in 1-2 words:\n\n{question}\n\nTopic:"
    topic_llm = Ollama(model="llama3.2")
    raw = topic_llm(topic_prompt)
    summary = raw.strip().split("\n")[0].replace("Topic:", "").strip(" \"'")
    return summary

# ---------------------------- TOOL SETUP ----------------------------
# ---------------------------- TOOL SETUP ----------------------------
search = wiki

tools = [search]

# ---------------------------- CHAT HISTORY ----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hello! I am your assistant John. I'm here to help you."}
    ]

# ---------------------------- DISPLAY CHAT ----------------------------
for msg in st.session_state.messages:
    role_class = "user" if msg["role"] == "user" else "assistant"
    st.markdown(f"<div class='chat-message {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

# ---------------------------- CHAT INPUT ----------------------------
if prompt := st.chat_input("Ask me anything..."):
    st.session_state["messages"].append({"role": "user", "content": f"👤 {prompt}"})
    st.markdown(f"<div class='chat-message user'>👤 {prompt}</div>", unsafe_allow_html=True)

    # Update topic if new question
    if prompt != st.session_state["last_topic_question"]:
        topic = summarize_topic(prompt)
        st.session_state["topic_summary"] = topic
        st.session_state["last_topic_question"] = prompt
        if topic not in st.session_state["previous_topics"]:
            st.session_state["previous_topics"].insert(0, topic)  # insert at top

    # LLM for assistant response
    llm = Ollama(model="llama3.2")

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            # response = agent.run(st.session_state.messages, callbacks=[st_cb])
            response = agent.run(prompt, callbacks=[st_cb])
            response = f"🤖 {response}"
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.markdown(f"<div class='chat-message assistant'>{response}</div>", unsafe_allow_html=True)

# ---------------------------- SIDEBAR TOPIC DISPLAY ----------------------------
if st.session_state["topic_summary"]:
    st.sidebar.markdown(f"**🟢 {st.session_state['topic_summary']}**")
else:
    st.sidebar.markdown("_No topic yet. Ask a question!_")

if st.session_state["previous_topics"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Previous Topics")
    for topic in st.session_state["previous_topics"]:
        st.sidebar.markdown(f"- {topic}")

st.sidebar.markdown("---")
st.sidebar.info("Ask a question below to start the session.")
