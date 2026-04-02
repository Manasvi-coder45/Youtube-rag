import os
import re
import requests
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TubeIQ — YouTube RAG Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e8eaf0;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ff4d6d22, #7c3aed22);
    border: 1px solid #ff4d6d55;
    color: #ff4d6d;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #ff4d6d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.6rem;
}
.hero-sub {
    color: #7a7f9a;
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 520px;
    margin: 0 auto 2rem;
    line-height: 1.7;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #12141c !important;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] .sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8eaf0;
    margin-bottom: 0.3rem;
}
[data-testid="stSidebar"] .sidebar-sub {
    font-size: 0.78rem;
    color: #555975;
    margin-bottom: 1.5rem;
    line-height: 1.5;
}

/* ── Input ── */
[data-testid="stTextInput"] input {
    background: #12141c !important;
    border: 1px solid #1e2130 !important;
    border-radius: 12px !important;
    color: #e8eaf0 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px #7c3aed22 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.4rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Pipeline card ── */
.pipeline-card {
    background: #12141c;
    border: 1px solid #1e2130;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.pipeline-card h3 {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #555975;
    margin-bottom: 1rem;
}
.step-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.42rem 0;
    font-size: 0.88rem;
    color: #7a7f9a;
    border-bottom: 1px solid #1a1d28;
}
.step-row:last-child { border-bottom: none; }
.step-row.done { color: #e8eaf0; }
.step-row.running { color: #a78bfa; }
.step-icon { font-size: 1rem; width: 1.4rem; text-align: center; }

/* ── Video card ── */
.video-card {
    background: #12141c;
    border: 1px solid #1e2130;
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.video-card img { width: 100%; display: block; }
.video-meta { padding: 0.9rem 1.1rem; }
.video-id-badge {
    display: inline-block;
    background: #1e2130;
    color: #7a7f9a;
    font-size: 0.75rem;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-family: monospace;
}

/* ── Answer & Summary boxes ── */
.answer-box {
    background: linear-gradient(135deg, #13162199, #1a1d2e99);
    border: 1px solid #2a2d45;
    border-left: 3px solid #7c3aed;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    font-size: 0.93rem;
    line-height: 1.8;
    color: #d4d7e8;
    white-space: pre-wrap;
}
.summary-box {
    background: linear-gradient(135deg, #13161f99, #1a1f2e99);
    border: 1px solid #2a3045;
    border-left: 3px solid #ff4d6d;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    font-size: 0.93rem;
    line-height: 1.8;
    color: #d4d7e8;
    white-space: pre-wrap;
}
.quiz-box {
    background: linear-gradient(135deg, #131a1699, #1a2a1e99);
    border: 1px solid #2a4535;
    border-left: 3px solid #22c55e;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    font-size: 0.93rem;
    line-height: 1.8;
    color: #d4d7e8;
    white-space: pre-wrap;
}

/* ── Chat messages ── */
.chat-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.6rem 0;
}
.chat-user-bubble {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.1rem;
    max-width: 72%;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-assistant {
    display: flex;
    justify-content: flex-start;
    margin: 0.6rem 0;
}
.chat-assistant-bubble {
    background: #1a1d2e;
    border: 1px solid #2a2d45;
    color: #d4d7e8;
    border-radius: 18px 18px 18px 4px;
    padding: 0.75rem 1.1rem;
    max-width: 82%;
    font-size: 0.9rem;
    line-height: 1.6;
    white-space: pre-wrap;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #e8eaf0;
    margin: 1.4rem 0 0.7rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Status pills ── */
.pill-success {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #14271e;
    border: 1px solid #22c55e44;
    color: #22c55e;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.35rem 0.85rem;
    border-radius: 100px;
}
.pill-info {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #141b27;
    border: 1px solid #3b82f644;
    color: #60a5fa;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.35rem 0.85rem;
    border-radius: 100px;
}

/* ── Divider ── */
hr { border-color: #1e2130 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #7c3aed !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #7a7f9a !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e8eaf0 !important;
    border-bottom-color: #7c3aed !important;
}

/* ── Transcript text area ── */
textarea {
    background: #12141c !important;
    border: 1px solid #1e2130 !important;
    color: #7a7f9a !important;
    border-radius: 12px !important;
    font-size: 0.82rem !important;
    font-family: monospace !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: #1e2130 !important;
    color: #a78bfa !important;
    border: 1px solid #2a2d45 !important;
    font-size: 0.85rem !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #2a2d45 !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ Powered by Groq + LangChain</div>
    <div class="hero-title">TubeIQ</div>
    <div class="hero-sub">Ask anything about any YouTube video. Get instant answers with timestamp citations.</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
steps = [
    "Fetch Transcript",
    "Build Document",
    "Chunk & Index",
    "Create Embeddings",
    "Build Vector DB",
    "Initialize Chain",
]
for key, default in [
    ("step_status", {s: "pending" for s in steps}),
    ("chain", None),
    ("transcript_text", None),
    ("video_id", None),
    ("chat_history", []),
    ("retriever", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url.strip()

def get_transcript(video_id: str):
    try:
        from youtube_transcript_api.proxies import GenericProxyConfig
        scraper_api_key = os.environ.get("SCRAPER_API_KEY")
        proxy_url = f"http://scraperapi:{scraper_api_key}@proxy-server.scraperapi.com:8001"
        proxy_config = GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
        session = requests.Session()
        session.verify = False
        fetched = YouTubeTranscriptApi(
            proxy_config=proxy_config, http_client=session
        ).fetch(video_id, languages=["en"])
        return fetched.to_raw_data()
    except TranscriptsDisabled:
        st.error("No captions available for this video.")
        return None
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

def build_document(transcript_list):
    full_text, offset_map, cursor = "", [], 0
    for chunk in transcript_list:
        text = chunk["text"] + " "
        offset_map.append({
            "char_start": cursor,
            "char_end": cursor + len(text),
            "start_time": chunk["start"],
            "end_time": chunk["start"] + chunk["duration"],
        })
        full_text += text
        cursor += len(text)
    return Document(page_content=full_text, metadata={"offset_map": offset_map}), full_text

def split_with_timestamps(doc):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    chunks = splitter.split_documents([doc])
    offset_map = doc.metadata["offset_map"]
    final_docs = []
    for chunk in chunks:
        s, e = chunk.metadata["start_index"], chunk.metadata["start_index"] + len(chunk.page_content)
        start_t, end_t = None, None
        for o in offset_map:
            if o["char_end"] >= s and start_t is None:
                start_t = o["start_time"]
            if o["char_start"] <= e:
                end_t = o["end_time"]
        chunk.metadata = {"start": start_t, "end": end_t}
        final_docs.append(chunk)
    return final_docs

@st.cache_resource(show_spinner=False)
def build_vector_store(_key, docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"Chunk {i}\nTranscript:\n{doc.page_content}\nStart: {doc.metadata.get('start')} | End: {doc.metadata.get('end')}")
    return "\n\n".join(parts)

def build_chain(retriever):
    llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0.2, streaming=True)
    prompt = PromptTemplate(
        template="""You are TubeIQ, an assistant that answers questions strictly from the YouTube video transcript provided.

Guidelines:
- Use ONLY the provided context.
- Always cite timestamps used in your answer.
- If the answer is not in the context, say: "I don't know based on the provided transcript."

Context:
{context}

Question:
{question}

Format:
Answer: <clear explanation>
Sources:
- Timestamp: <start_time - end_time>
""",
        input_variables=["context", "question"],
    )
    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        })
        | prompt | llm | StrOutputParser()
    )
    return chain

def extract_earliest_timestamp(text):
    matches = re.findall(r"Timestamp:\s*([\d\.]+)\s*-\s*([\d\.]+)", text)
    if not matches:
        return 0
    return int(min(float(s) for s, _ in matches))

def stream_response(prompt_text, box_class="answer-box"):
    box = st.empty()
    full = ""
    for chunk in st.session_state.chain.stream(prompt_text):
        full += chunk
        box.markdown(f'<div class="{box_class}">{full}▌</div>', unsafe_allow_html=True)
    box.markdown(f'<div class="{box_class}">{full}</div>', unsafe_allow_html=True)
    return full

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🎬 TubeIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Paste any YouTube URL to start asking questions, get summaries, quizzes and more.</div>', unsafe_allow_html=True)

    video_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")

    if st.session_state.chain is not None:
        st.markdown("---")
        st.markdown('<div class="pill-success">✦ Video ready</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Load new video"):
            for key in ["chain", "transcript_text", "video_id", "chat_history", "retriever"]:
                st.session_state[key] = None if key != "chat_history" else []
            st.session_state.step_status = {s: "pending" for s in steps}
            st.rerun()

    st.markdown("---")
    st.markdown("**Pipeline**")

    # Render pipeline steps in sidebar
    icons = {"pending": "⏳", "running": "🔄", "done": "✅"}
    for step in steps:
        status = st.session_state.step_status[step]
        css = "done" if status == "done" else ("running" if status == "running" else "")
        st.markdown(
            f'<div class="step-row {css}"><span class="step-icon">{icons[status]}</span>{step}</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
if not video_url:
    # Landing state
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "💬", "Ask Anything", "Ask questions in plain English and get cited answers from the transcript."),
        (c2, "📋", "Auto Summarize", "Get a structured summary of any video with key points and timestamps."),
        (c3, "🧠", "Quiz Generator", "Turn any video into a set of quiz questions to test your understanding."),
    ]:
        with col:
            st.markdown(f"""
            <div class="pipeline-card" style="text-align:center;padding:2rem 1.2rem;">
                <div style="font-size:2rem;margin-bottom:0.6rem">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;margin-bottom:0.4rem">{title}</div>
                <div style="color:#555975;font-size:0.85rem;line-height:1.6">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

# ─── Extract video ID ───
video_id = extract_video_id(video_url)

# ─── Thumbnail + meta ───
col_thumb, col_right = st.columns([1, 2], gap="large")
with col_thumb:
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
    st.markdown(f"""
    <div class="video-card">
        <img src="{thumbnail_url}" />
        <div class="video-meta">
            <span class="video-id-badge">ID: {video_id}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Pipeline processing ───
with col_right:
    if st.session_state.chain is None:

        def set_step(step, status):
            st.session_state.step_status[step] = status

        progress_placeholder = st.empty()

        def show_progress(msg):
            progress_placeholder.markdown(f'<div class="pill-info">⟳ {msg}</div>', unsafe_allow_html=True)

        # Step 1
        set_step("Fetch Transcript", "running")
        show_progress("Fetching transcript...")
        transcript_list = get_transcript(video_id)
        if not transcript_list:
            st.stop()
        set_step("Fetch Transcript", "done")

        # Step 2
        set_step("Build Document", "running")
        show_progress("Building document...")
        doc, full_text = build_document(transcript_list)
        st.session_state.transcript_text = full_text
        set_step("Build Document", "done")

        # Step 3
        set_step("Chunk & Index", "running")
        show_progress("Chunking transcript...")
        final_docs = split_with_timestamps(doc)
        set_step("Chunk & Index", "done")

        # Step 4 + 5
        set_step("Create Embeddings", "running")
        show_progress("Creating embeddings & building vector DB...")
        set_step("Create Embeddings", "done")
        set_step("Build Vector DB", "running")
        vector_store = build_vector_store(video_id, final_docs)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        st.session_state.retriever = retriever
        set_step("Build Vector DB", "done")

        # Step 6
        set_step("Initialize Chain", "running")
        show_progress("Initializing RAG chain...")
        st.session_state.chain = build_chain(retriever)
        st.session_state.video_id = video_id
        set_step("Initialize Chain", "done")

        progress_placeholder.markdown('<div class="pill-success">✦ Ready — ask your first question below</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="pill-success">✦ Video loaded and ready</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick action buttons ──
    b1, b2, b3 = st.columns(3)
    with b1:
        summarize = st.button("📋 Summarize")
    with b2:
        quiz = st.button("🧠 Generate Quiz")
    with b3:
        keypoints = st.button("🔑 Key Points")

# ─────────────────────────────────────────────
# TABS: Chat | Tools | Transcript
# ─────────────────────────────────────────────
tab_chat, tab_tools, tab_transcript = st.tabs(["💬 Chat", "🛠 Tools", "📄 Transcript"])

# ── TAB 1: Chat ──
with tab_chat:
    # Render history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user"><div class="chat-user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant"><div class="chat-assistant-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)

    question = st.text_input("Ask a question about the video...", key="chat_input", label_visibility="collapsed")

    if question and st.session_state.chain:
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.markdown(f'<div class="chat-user"><div class="chat-user-bubble">{question}</div></div>', unsafe_allow_html=True)

        with st.spinner(""):
            box = st.empty()
            full = ""
            for chunk in st.session_state.chain.stream(question):
                full += chunk
                box.markdown(f'<div class="chat-assistant"><div class="chat-assistant-bubble">{full}▌</div></div>', unsafe_allow_html=True)
            box.markdown(f'<div class="chat-assistant"><div class="chat-assistant-bubble">{full}</div></div>', unsafe_allow_html=True)

        st.session_state.chat_history.append({"role": "assistant", "content": full})

        # Timestamp video seek
        earliest = extract_earliest_timestamp(full)
        if earliest > 0:
            st.markdown('<div class="section-header">🎯 Relevant Video Segment</div>', unsafe_allow_html=True)
            embed_url = f"https://www.youtube.com/embed/{st.session_state.video_id}?start={earliest}&autoplay=1"
            st.components.v1.iframe(embed_url, height=320)
            st.caption(f"Jumping to {earliest}s based on cited timestamps")

    if st.session_state.chat_history:
        if st.button("🗑 Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

# ── TAB 2: Tools ──
with tab_tools:
    if st.session_state.chain is None:
        st.info("Load a video first to use the tools.")
    else:
        # Summarize
        if summarize:
            st.markdown('<div class="section-header">📋 Video Summary</div>', unsafe_allow_html=True)
            with st.spinner("Generating summary..."):
                stream_response(
                    "Give me a well-structured summary of this video covering: 1) Main topic and purpose, 2) Key points and arguments made, 3) Important conclusions or takeaways, 4) Notable timestamps to revisit.",
                    "summary-box"
                )

        # Quiz
        if quiz:
            st.markdown('<div class="section-header">🧠 Quiz Questions</div>', unsafe_allow_html=True)
            with st.spinner("Generating quiz..."):
                stream_response(
                    "Generate 5 multiple choice quiz questions based on the content of this video. For each question provide: the question, 4 options (A/B/C/D), and the correct answer with a brief explanation referencing the transcript timestamp.",
                    "quiz-box"
                )

        # Key Points
        if keypoints:
            st.markdown('<div class="section-header">🔑 Key Points</div>', unsafe_allow_html=True)
            with st.spinner("Extracting key points..."):
                stream_response(
                    "Extract the top 7 most important key points from this video. Format each as a concise bullet point with the relevant timestamp range.",
                    "summary-box"
                )

        st.markdown("---")

        # Keyword search
        st.markdown('<div class="section-header">🔍 Keyword Search in Transcript</div>', unsafe_allow_html=True)
        keyword = st.text_input("Search for a word or phrase in the transcript", key="keyword_search")
        if keyword and st.session_state.transcript_text:
            text = st.session_state.transcript_text
            count = text.lower().count(keyword.lower())
            if count:
                st.markdown(f'<div class="pill-success">✦ Found "{keyword}" {count} time(s)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pill-info">No occurrences of "{keyword}" found.</div>', unsafe_allow_html=True)

# ── TAB 3: Transcript ──
with tab_transcript:
    if st.session_state.transcript_text:
        st.markdown('<div class="section-header">📄 Full Transcript</div>', unsafe_allow_html=True)
        word_count = len(st.session_state.transcript_text.split())
        char_count = len(st.session_state.transcript_text)
        mc1, mc2 = st.columns(2)
        mc1.metric("Words", f"{word_count:,}")
        mc2.metric("Characters", f"{char_count:,}")
        st.text_area("", st.session_state.transcript_text, height=400, label_visibility="collapsed")
        st.download_button(
            label="⬇ Download transcript (.txt)",
            data=st.session_state.transcript_text,
            file_name=f"transcript_{st.session_state.video_id}.txt",
            mime="text/plain",
        )
    else:
        st.info("Load a video to view its transcript here.")