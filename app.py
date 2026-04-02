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

# ════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════
st.set_page_config(
    page_title="TubeIQ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #080a0f;
    --surface:   #0e1117;
    --surface2:  #141720;
    --border:    #1c1f2e;
    --border2:   #252840;
    --text:      #e2e5f0;
    --muted:     #555a78;
    --muted2:    #363a52;
    --accent:    #6366f1;
    --accent2:   #818cf8;
    --red:       #f43f5e;
    --green:     #10b981;
    --amber:     #f59e0b;
    --font:      'Outfit', sans-serif;
    --mono:      'JetBrains Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem; max-width: 1380px; margin: 0 auto; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

/* Hero */
.hero-wrap { padding: 3rem 0 2rem; text-align: center; }
.hero-eyebrow {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--accent2); margin-bottom: 1rem;
}
.hero-title {
    font-size: clamp(3rem, 6vw, 5rem); font-weight: 800; line-height: 1;
    letter-spacing: -0.03em;
    background: linear-gradient(160deg, #fff 20%, var(--accent2) 60%, var(--red) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 0.7rem;
}
.hero-sub {
    font-size: 1rem; font-weight: 300; color: var(--muted);
    max-width: 440px; margin: 0 auto; line-height: 1.7;
}

/* Sidebar */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem; }

/* Inputs */
[data-testid="stTextInput"] input {
    background: var(--surface2) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text) !important;
    font-family: var(--font) !important; font-size: 0.9rem !important;
    padding: 0.7rem 1rem !important; transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
[data-testid="stTextInput"] label { color: var(--muted) !important; font-size: 0.8rem !important; font-weight: 500 !important; }

/* Buttons */
.stButton > button {
    background: var(--surface2) !important; color: var(--text) !important;
    border: 1px solid var(--border2) !important; border-radius: 8px !important;
    font-family: var(--font) !important; font-weight: 500 !important;
    font-size: 0.85rem !important; padding: 0.55rem 1rem !important;
    transition: all 0.18s ease !important; width: 100%;
}
.stButton > button:hover {
    background: var(--border2) !important; border-color: var(--accent) !important;
    color: var(--accent2) !important; transform: translateY(-1px) !important;
}

/* Tabs */
[data-testid="stTabs"] { border-bottom: 1px solid var(--border); }
[data-testid="stTabs"] button {
    font-family: var(--font) !important; font-size: 0.88rem !important;
    font-weight: 500 !important; color: var(--muted) !important;
    padding: 0.6rem 1.2rem !important; border-radius: 0 !important;
    border: none !important; background: transparent !important;
}
[data-testid="stTabs"] button:hover { color: var(--text) !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--text) !important; border-bottom: 2px solid var(--accent) !important;
}
[data-testid="stTabs"] [data-testid="stTabsContent"] { padding-top: 1.5rem; }

/* Cards */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.4rem; margin-bottom: 0.8rem; }

/* Thumbnail */
.thumb-wrap { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); background: var(--surface); }
.thumb-wrap img { width: 100%; display: block; }
.thumb-meta { padding: 0.8rem 1rem; }
.vid-badge { font-family: var(--mono); font-size: 0.72rem; background: var(--surface2); border: 1px solid var(--border2); color: var(--muted); padding: 0.2rem 0.6rem; border-radius: 5px; }

/* Pills */
.pill { display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.75rem; font-weight: 500; padding: 0.28rem 0.75rem; border-radius: 100px; }
.pill-green { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.25); color: var(--green); }
.pill-blue  { background: rgba(99,102,241,0.1);  border: 1px solid rgba(99,102,241,0.3);  color: var(--accent2); }
.pill-red   { background: rgba(244,63,94,0.1);   border: 1px solid rgba(244,63,94,0.25);  color: var(--red); }

/* Steps */
.step-item { display: flex; align-items: center; gap: 0.6rem; padding: 0.38rem 0.5rem; font-size: 0.82rem; color: var(--muted2); border-radius: 7px; }
.step-item.done    { color: var(--green); }
.step-item.running { color: var(--accent2); background: rgba(99,102,241,0.06); }
.step-icon { width: 1.1rem; text-align: center; font-size: 0.85rem; }

/* Chat bubbles */
.chat-row-user  { display: flex; justify-content: flex-end; margin: 0.5rem 0; animation: fadeUp 0.2s ease; }
.chat-row-bot   { display: flex; justify-content: flex-start; margin: 0.5rem 0; animation: fadeUp 0.2s ease; }
.bubble-user {
    background: linear-gradient(135deg, var(--accent), #7c3aed);
    color: #fff; padding: 0.7rem 1.1rem;
    border-radius: 18px 18px 4px 18px; max-width: 75%;
    font-size: 0.88rem; line-height: 1.6; box-shadow: 0 2px 12px rgba(99,102,241,0.25);
}
.bubble-bot {
    background: var(--surface2); border: 1px solid var(--border2); color: var(--text);
    padding: 0.7rem 1.1rem; border-radius: 18px 18px 18px 4px; max-width: 82%;
    font-size: 0.88rem; line-height: 1.7; white-space: pre-wrap;
}
@keyframes fadeUp { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }

/* Output boxes */
.out-box { background: var(--surface2); border: 1px solid var(--border2); border-radius: 12px; padding: 1.3rem 1.5rem; font-size: 0.88rem; line-height: 1.8; color: var(--text); white-space: pre-wrap; }
.out-box.accent { border-left: 3px solid var(--accent); }
.out-box.red    { border-left: 3px solid var(--red); }
.out-box.green  { border-left: 3px solid var(--green); }

/* Quiz */
.quiz-q { font-weight: 600; font-size: 0.95rem; color: var(--text); margin-bottom: 0.8rem; line-height: 1.5; }
.quiz-opt { display: flex; align-items: center; gap: 0.7rem; padding: 0.65rem 1rem; margin-bottom: 0.4rem; border: 1px solid var(--border2); border-radius: 9px; background: var(--surface2); font-size: 0.87rem; }
.quiz-opt.correct { border-color: var(--green) !important; background: rgba(16,185,129,0.1) !important; color: var(--green) !important; }
.quiz-opt.wrong   { border-color: var(--red)   !important; background: rgba(244,63,94,0.1)   !important; color: var(--red)   !important; }
.quiz-opt.reveal  { border-color: var(--accent) !important; background: rgba(99,102,241,0.07) !important; }
.quiz-score { font-family: var(--font); font-size: 1rem; font-weight: 600; padding: 1rem 1.4rem; background: var(--surface2); border: 1px solid var(--border2); border-radius: 12px; margin-top: 1rem; text-align: center; }

/* Section label */
.sec-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.7rem; }

/* Metrics */
[data-testid="stMetric"] { background: var(--surface2); border: 1px solid var(--border); border-radius: 10px; padding: 0.8rem 1rem; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.4rem !important; font-weight: 700 !important; }

/* Download */
[data-testid="stDownloadButton"] button { background: var(--surface2) !important; color: var(--accent2) !important; border: 1px solid var(--border2) !important; }

/* Misc */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
textarea { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.78rem !important; }

/* Sidebar labels */
.sb-title { font-size: 1.1rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; margin-bottom: 0.2rem; }
.sb-sub   { font-size: 0.78rem; color: var(--muted); line-height: 1.5; margin-bottom: 1.4rem; }

/* Feature cards */
.feat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.8rem 1.4rem; text-align: center; transition: border-color 0.2s, transform 0.2s; }
.feat-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.feat-icon  { font-size: 2rem; margin-bottom: 0.7rem; }
.feat-title { font-weight: 700; font-size: 0.95rem; margin-bottom: 0.4rem; color: var(--text); }
.feat-desc  { font-size: 0.82rem; color: var(--muted); line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════
STEPS = ["Fetch Transcript", "Build Document", "Chunk & Index",
         "Create Embeddings", "Build Vector DB", "Initialize Chain"]

_defaults: dict = {
    "step_status":     {s: "pending" for s in STEPS},
    "chain":           None,
    "transcript_text": None,
    "video_id":        None,
    "chat_history":    [],
    "quiz_questions":  [],
    "quiz_answers":    {},
    "quiz_submitted":  {},
    "quiz_generated":  False,
    "summary_text":    None,
    "keypoints_text":  None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════

def extract_video_id(url: str) -> str:
    url = url.strip()
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url


def get_transcript(video_id: str):
    try:
        from youtube_transcript_api.proxies import GenericProxyConfig
        scraper_api_key = os.environ.get("SCRAPER_API_KEY", "")
        proxy_url = f"http://scraperapi:{scraper_api_key}@proxy-server.scraperapi.com:8001"
        proxy_config = GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
        session = requests.Session()
        session.verify = False
        requests.packages.urllib3.disable_warnings()
        api = YouTubeTranscriptApi(proxy_config=proxy_config, http_client=session)
        try:
            fetched = api.fetch(video_id, languages=["en"])
        except Exception:
            fetched = api.fetch(video_id)
        return fetched.to_raw_data()
    except TranscriptsDisabled:
        st.error("No captions available for this video.")
        return None
    except Exception as e:
        st.error(f"Transcript error: {e}")
        return None


def build_document(transcript_list):
    full_text, offset_map, cursor = "", [], 0
    for chunk in transcript_list:
        text = chunk["text"].strip() + " "
        offset_map.append({
            "char_start": cursor,
            "char_end":   cursor + len(text),
            "start_time": chunk["start"],
            "end_time":   chunk["start"] + chunk["duration"],
        })
        full_text += text
        cursor += len(text)
    doc = Document(page_content=full_text, metadata={"offset_map": offset_map})
    return doc, full_text


def split_with_timestamps(doc):
    total = len(doc.page_content)
    chunk_size    = 1500 if total > 100_000 else 1000
    chunk_overlap = 300  if total > 100_000 else 200
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents([doc])
    offset_map = doc.metadata["offset_map"]
    final_docs = []
    for chunk in chunks:
        s = chunk.metadata["start_index"]
        e = s + len(chunk.page_content)
        start_t = end_t = None
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
    for i, d in enumerate(docs, 1):
        parts.append(
            f"[Chunk {i}] Start={d.metadata.get('start','?')}s End={d.metadata.get('end','?')}s\n{d.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def build_chain(retriever):
    llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", temperature=0.2, streaming=True)
    prompt = PromptTemplate(
        template="""You are TubeIQ, an AI assistant. Answer ONLY from the transcript context below.

Rules:
- Use only the provided context. Do not hallucinate.
- Always cite timestamp ranges.
- If not in context say: "I don't know based on the provided transcript."

Transcript context:
{context}

Question: {question}

Format:
Answer: <explanation>

Sources:
- Timestamp: <start - end>
""",
        input_variables=["context", "question"],
    )
    return (
        RunnableParallel({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()})
        | prompt | llm | StrOutputParser()
    )


def extract_earliest_timestamp(text: str) -> int:
    matches = re.findall(r"Timestamp:\s*([\d\.]+)\s*-\s*([\d\.]+)", text)
    return int(min(float(s) for s, _ in matches)) if matches else 0


def stream_to_box(prompt_text: str, css_class: str = "accent") -> str:
    box = st.empty()
    full = ""
    for chunk in st.session_state.chain.stream(prompt_text):
        full += chunk
        box.markdown(f'<div class="out-box {css_class}">{full}▌</div>', unsafe_allow_html=True)
    box.markdown(f'<div class="out-box {css_class}">{full}</div>', unsafe_allow_html=True)
    return full


def parse_quiz(raw: str) -> list:
    questions = []
    blocks = re.split(r"\n?Q\d+[\.\)]\s*", raw.strip())
    for block in blocks:
        if not block.strip():
            continue
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue
        q_text = lines[0]
        options = {}
        correct = None
        for line in lines[1:]:
            m = re.match(r"^([A-D])[\)\.]\s*(.+)", line)
            if m:
                options[m.group(1)] = m.group(2)
            ans_m = re.match(r"(?:Answer|Correct)[:\s]+([A-D])", line, re.IGNORECASE)
            if ans_m:
                correct = ans_m.group(1).upper()
        if q_text and len(options) >= 2 and correct:
            questions.append({"q": q_text, "options": options, "answer": correct})
    return questions


# ════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-title">🎬 TubeIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">AI-powered video intelligence. Chat, summarize, quiz.</div>', unsafe_allow_html=True)

    video_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.chain:
        st.markdown('<span class="pill pill-green">● Video Ready</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺  Load new video"):
            for k, v in _defaults.items():
                st.session_state[k] = v
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Pipeline</div>', unsafe_allow_html=True)
    icons = {"pending": "○", "running": "◌", "done": "●"}
    for step in STEPS:
        status = st.session_state.step_status[step]
        st.markdown(
            f'<div class="step-item {status}"><span class="step-icon">{icons[status]}</span>{step}</div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">Powered by Groq · LangChain · FAISS</div>
    <div class="hero-title">TubeIQ</div>
    <div class="hero-sub">Ask anything about any YouTube video. Instant answers, summaries and interactive quizzes.</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# LANDING
# ════════════════════════════════════════════════════
if not video_url:
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "💬", "Smart Chat",       "Ask follow-up questions and get cited answers from the transcript."),
        (c2, "📋", "Auto Summary",     "Structured summaries with key points and timestamps in seconds."),
        (c3, "🧠", "Interactive Quiz", "Take an auto-generated quiz and get instant feedback per answer."),
        (c4, "📄", "Transcript",       "View, search and download the full transcript of any video."),
    ]:
        with col:
            st.markdown(f"""
            <div class="feat-card">
                <div class="feat-icon">{icon}</div>
                <div class="feat-title">{title}</div>
                <div class="feat-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════
# EXTRACT VIDEO ID + RESET ON URL CHANGE
# ════════════════════════════════════════════════════
video_id = extract_video_id(video_url)

if st.session_state.video_id and st.session_state.video_id != video_id:
    for k, v in _defaults.items():
        st.session_state[k] = v
    st.rerun()


# ════════════════════════════════════════════════════
# TOP ROW — thumbnail + pipeline
# ════════════════════════════════════════════════════
col_thumb, col_main = st.columns([5, 7], gap="large")

with col_thumb:
    st.markdown(f"""
    <div class="thumb-wrap">
        <img src="https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
             onerror="this.src='https://img.youtube.com/vi/{video_id}/hqdefault.jpg'"/>
        <div class="thumb-meta"><span class="vid-badge">{video_id}</span></div>
    </div>""", unsafe_allow_html=True)

with col_main:
    progress_slot = st.empty()

    if st.session_state.chain is None:
        def mark(step, status):
            st.session_state.step_status[step] = status

        def info(msg):
            progress_slot.markdown(f'<span class="pill pill-blue">⟳ {msg}</span>', unsafe_allow_html=True)

        mark("Fetch Transcript", "running"); info("Fetching transcript…")
        transcript_list = get_transcript(video_id)
        if not transcript_list:
            st.stop()
        mark("Fetch Transcript", "done")

        mark("Build Document", "running"); info("Building document…")
        doc, full_text = build_document(transcript_list)
        st.session_state.transcript_text = full_text
        mark("Build Document", "done")

        mark("Chunk & Index", "running"); info("Chunking transcript…")
        final_docs = split_with_timestamps(doc)
        mark("Chunk & Index", "done")

        mark("Create Embeddings", "running"); info("Creating embeddings…")
        mark("Create Embeddings", "done")
        mark("Build Vector DB", "running"); info("Building vector database…")
        vector_store = build_vector_store(video_id, final_docs)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        mark("Build Vector DB", "done")

        mark("Initialize Chain", "running"); info("Initializing chain…")
        st.session_state.chain    = build_chain(retriever)
        st.session_state.video_id = video_id
        mark("Initialize Chain", "done")

        progress_slot.markdown('<span class="pill pill-green">✓ Ready — use the tabs below</span>', unsafe_allow_html=True)
    else:
        progress_slot.markdown('<span class="pill pill-green">✓ Video loaded</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick action buttons
    qa1, qa2, qa3 = st.columns(3)
    with qa1: do_summary   = st.button("📋 Summarize",  key="qa_sum")
    with qa2: do_quiz      = st.button("🧠 Quiz",       key="qa_quiz")
    with qa3: do_keypoints = st.button("🔑 Key Points", key="qa_kp")


# ════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════
tab_chat, tab_tools, tab_quiz, tab_transcript = st.tabs(
    ["💬 Chat", "🛠 Tools", "🧠 Quiz", "📄 Transcript"]
)


# ────────────────────────────────────────────────────
# TAB 1 — CHAT  (form prevents double-submit loop)
# ────────────────────────────────────────────────────
with tab_chat:
    # Render stored history first
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-row-user"><div class="bubble-user">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-row-bot"><div class="bubble-bot">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )

    # Form submits cleanly on Enter and clears input — prevents looping
    with st.form(key="chat_form", clear_on_submit=True):
        fc1, fc2 = st.columns([8, 1])
        with fc1:
            question = st.text_input(
                "q", placeholder="Ask anything about this video…", label_visibility="collapsed"
            )
        with fc2:
            submitted = st.form_submit_button("➤")

    if submitted and question.strip():
        q = question.strip()
        st.session_state.chat_history.append({"role": "user", "content": q})
        st.markdown(
            f'<div class="chat-row-user"><div class="bubble-user">{q}</div></div>',
            unsafe_allow_html=True,
        )

        with st.spinner(""):
            box  = st.empty()
            full = ""
            for chunk in st.session_state.chain.stream(q):
                full += chunk
                box.markdown(
                    f'<div class="chat-row-bot"><div class="bubble-bot">{full}▌</div></div>',
                    unsafe_allow_html=True,
                )
            box.markdown(
                f'<div class="chat-row-bot"><div class="bubble-bot">{full}</div></div>',
                unsafe_allow_html=True,
            )

        st.session_state.chat_history.append({"role": "assistant", "content": full})

        earliest = extract_earliest_timestamp(full)
        if earliest > 0:
            st.markdown('<div class="sec-label" style="margin-top:1rem">🎯 Relevant segment</div>', unsafe_allow_html=True)
            st.components.v1.iframe(
                f"https://www.youtube.com/embed/{st.session_state.video_id}?start={earliest}",
                height=300,
            )

    if st.session_state.chat_history:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ────────────────────────────────────────────────────
# TAB 2 — TOOLS
# ────────────────────────────────────────────────────
with tab_tools:
    t1, t2 = st.columns(2)
    with t1: trig_summary   = st.button("📋 Summarize",  key="t_sum")
    with t2: trig_keypoints = st.button("🔑 Key Points", key="t_kp")

    run_summary   = do_summary   or trig_summary
    run_keypoints = do_keypoints or trig_keypoints

    if run_summary:
        st.markdown('<div class="sec-label">📋 Summary</div>', unsafe_allow_html=True)
        with st.spinner("Generating summary…"):
            result = stream_to_box(
                "Provide a well-structured summary covering: "
                "1) Main topic and purpose, 2) Key arguments and ideas, "
                "3) Important takeaways, 4) Notable timestamps to revisit.",
                "accent",
            )
            st.session_state.summary_text = result
    elif st.session_state.summary_text:
        st.markdown('<div class="sec-label">📋 Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="out-box accent">{st.session_state.summary_text}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    if run_keypoints:
        st.markdown('<div class="sec-label">🔑 Key Points</div>', unsafe_allow_html=True)
        with st.spinner("Extracting key points…"):
            result = stream_to_box(
                "List the 8 most important key points from this video. "
                "One concise bullet per point with the relevant timestamp range.",
                "red",
            )
            st.session_state.keypoints_text = result
    elif st.session_state.keypoints_text:
        st.markdown('<div class="sec-label">🔑 Key Points</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="out-box red">{st.session_state.keypoints_text}</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">🔍 Keyword search</div>', unsafe_allow_html=True)
    kw = st.text_input("Keyword", placeholder="e.g. neural network", key="kw_search")
    if kw and st.session_state.transcript_text:
        count = st.session_state.transcript_text.lower().count(kw.lower())
        if count:
            st.markdown(f'<span class="pill pill-green">✓ "{kw}" found {count} time(s)</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="pill pill-red">✗ "{kw}" not found</span>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────
# TAB 3 — INTERACTIVE QUIZ
# ────────────────────────────────────────────────────
with tab_quiz:
    gen_col, _ = st.columns([2, 5])
    with gen_col:
        trig_gen = st.button("🧠 Generate quiz", key="q_gen")

    if do_quiz or trig_gen:
        # Reset quiz state
        st.session_state.quiz_questions = []
        st.session_state.quiz_answers   = {}
        st.session_state.quiz_submitted = {}
        st.session_state.quiz_generated = False

        with st.spinner("Generating quiz…"):
            raw_quiz = ""
            box = st.empty()
            for chunk in st.session_state.chain.stream(
                "Generate exactly 5 multiple choice questions from the video transcript.\n"
                "Use this EXACT format for every question:\n\n"
                "Q1. <question text>\n"
                "A) <option>\nB) <option>\nC) <option>\nD) <option>\n"
                "Answer: <letter>\n\n"
                "Q2. ... and so on. Make questions varied in difficulty."
            ):
                raw_quiz += chunk
                box.markdown(f'<div class="out-box accent">{raw_quiz}▌</div>', unsafe_allow_html=True)
            box.empty()

        parsed = parse_quiz(raw_quiz)
        if parsed:
            st.session_state.quiz_questions = parsed
            st.session_state.quiz_generated = True
        else:
            st.warning("Could not parse the quiz. Try generating again.")

    # ── Render quiz ──
    if st.session_state.quiz_generated and st.session_state.quiz_questions:
        questions = st.session_state.quiz_questions
        total     = len(questions)

        st.markdown('<div class="sec-label">Click an option to submit your answer</div>', unsafe_allow_html=True)

        correct_count = 0

        for i, q_data in enumerate(questions):
            q_text  = q_data["q"]
            options = q_data["options"]
            answer  = q_data["answer"]
            chosen  = st.session_state.quiz_answers.get(i)
            done    = st.session_state.quiz_submitted.get(i, False)

            st.markdown(f'<div class="quiz-q">Q{i+1}. {q_text}</div>', unsafe_allow_html=True)

            if not done:
                # Show clickable buttons for unanswered questions
                for letter, text in options.items():
                    if st.button(f"{letter})  {text}", key=f"q{i}_{letter}"):
                        st.session_state.quiz_answers[i]   = letter
                        st.session_state.quiz_submitted[i] = True
                        st.rerun()
            else:
                # Show styled result boxes
                for letter, text in options.items():
                    if letter == answer:
                        css = "correct"
                        icon = "✓"
                    elif letter == chosen:
                        css = "wrong"
                        icon = "✗"
                    else:
                        css = ""
                        icon = " "
                    st.markdown(
                        f'<div class="quiz-opt {css}">'
                        f'<strong style="min-width:1.2rem">{letter}</strong>'
                        f'<span style="margin-right:0.4rem">{icon}</span>{text}</div>',
                        unsafe_allow_html=True,
                    )

                if chosen == answer:
                    correct_count += 1
                    st.markdown('<span class="pill pill-green">✓ Correct!</span>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<span class="pill pill-red">✗ Correct answer: {answer}) {options.get(answer,"")}</span>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<hr>", unsafe_allow_html=True)

        # Score summary
        answered = len(st.session_state.quiz_submitted)
        if answered == total:
            pct   = int((correct_count / total) * 100)
            grade = "🏆 Excellent!" if pct >= 80 else ("👍 Good job!" if pct >= 60 else "📖 Keep studying!")
            st.markdown(f"""
            <div class="quiz-score">
                {grade} &nbsp;&nbsp; Score: <strong>{correct_count}/{total}</strong> ({pct}%)
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="pill pill-blue">{answered}/{total} answered</span>', unsafe_allow_html=True)

    elif not st.session_state.quiz_generated:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem 2rem">
            <div style="font-size:2.5rem;margin-bottom:0.8rem">🧠</div>
            <div style="font-weight:600;margin-bottom:0.4rem;color:var(--text)">Interactive Quiz</div>
            <div style="color:var(--muted);font-size:0.85rem;line-height:1.6">
                Click "Generate quiz" above to create 5 questions from this video.<br>
                Answer each question and get instant feedback.
            </div>
        </div>""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────
# TAB 4 — TRANSCRIPT
# ────────────────────────────────────────────────────
with tab_transcript:
    if st.session_state.transcript_text:
        txt       = st.session_state.transcript_text
        wc        = len(txt.split())
        cc        = len(txt)
        est_mins  = round(cc / 14 / 60)

        m1, m2, m3 = st.columns(3)
        m1.metric("Words",       f"{wc:,}")
        m2.metric("Characters",  f"{cc:,}")
        m3.metric("Est. length", f"~{est_mins} min")

        st.markdown("<br>", unsafe_allow_html=True)
        st.text_area("Full transcript", txt, height=420, label_visibility="visible")
        st.download_button(
            label="⬇ Download transcript (.txt)",
            data=txt,
            file_name=f"transcript_{st.session_state.video_id}.txt",
            mime="text/plain",
        )
    else:
        st.info("Load a video to view its full transcript here.")