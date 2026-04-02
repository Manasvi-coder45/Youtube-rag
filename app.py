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
# -------------------------------
# CONFIGURATION
# -------------------------------

# Put your Groq API key here
load_dotenv()

# -------------------------------
# STREAMLIT PAGE
# -------------------------------

st.set_page_config(page_title="YouTube RAG Assistant", layout="wide")



st.markdown("""
<style>

.main-title {
    font-size:42px;
    font-weight:700;
    text-align:center;
    margin-bottom:5px;
}

.sub-title {
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.pipeline-box {
    padding:20px;
    border-radius:12px;
    background:#f8f9fb;
    border:1px solid #e5e7eb;
}

.answer-box {
    padding:20px;
    border-radius:12px;
    background:#f0f7ff;
    border:1px solid #c7ddff;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">YouTube RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask questions about any YouTube video using transcript intelligence</div>', unsafe_allow_html=True)

url_container = st.container()
pipeline_container = st.container()
status_container = st.container()
qa_container = st.container()

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def get_transcript(video_id):

    try:
        fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        transcript_list = fetched_transcript.to_raw_data()

    except TranscriptsDisabled:
        st.error("No captions available for this video.")
        return None

    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

    return transcript_list


def build_document(transcript_list):

    full_transcript = ""
    offset_map = []

    cursor = 0

    for chunk in transcript_list:

        text = chunk["text"] + " "
        start = chunk["start"]
        end = chunk["start"] + chunk["duration"]

        full_transcript += text

        offset_map.append({
            "char_start": cursor,
            "char_end": cursor + len(text),
            "start_time": start,
            "end_time": end
        })

        cursor += len(text)

    doc = Document(
        page_content=full_transcript,
        metadata={"offset_map": offset_map}
    )

    return doc


def split_with_timestamps(doc):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    chunks = splitter.split_documents([doc])

    offset_map = doc.metadata["offset_map"]

    final_docs = []

    for chunk in chunks:

        start_char = chunk.metadata["start_index"]
        end_char = start_char + len(chunk.page_content)

        start_time = None
        end_time = None

        for o in offset_map:

            if o["char_end"] >= start_char and start_time is None:
                start_time = o["start_time"]

            if o["char_start"] <= end_char:
                end_time = o["end_time"]

        chunk.metadata = {
            "start": start_time,
            "end": end_time
        }

        final_docs.append(chunk)

    return final_docs


def build_vector_store(docs):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vector_store = FAISS.from_documents(docs, embeddings)

    return vector_store


def format_docs(retrieved_docs):

    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):

        start = doc.metadata.get("start", None)
        end = doc.metadata.get("end", None)

        context_parts.append(
            f"""
Chunk {i}

Transcript:
{doc.page_content}

Metadata
Start: {start}
End: {end}
"""
        )

    return "\n\n".join(context_parts)


# -------------------------------
# PROMPT TEMPLATE
# -------------------------------

prompt = PromptTemplate(
    template="""
You are an assistant that answers questions using the provided transcript context.

Guidelines:

- Use ONLY the provided context.
- Each context block contains metadata such as timestamp.
- If the answer is not contained in the context say:
"I don't know based on the provided transcript."
- Always cite the relevant metadata timestamps used to produce the answer.

Context:
{context}

Question:
{question}

Return the response in the following format:

Answer: <clear explanation>

Sources:

- Timestamp: <start_time - end_time>
""",
    input_variables=["context", "question"]
)


# -------------------------------
# BUILD RAG PIPELINE
# -------------------------------

def build_chain(retriever):

    llm = ChatGroq(
        model="moonshotai/kimi-k2-instruct-0905",
        temperature=0.2
    )

    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | parser

    return main_chain


# -------------------------------
# STREAMLIT UI
# -------------------------------

steps = [
    "Fetch Transcript",
    "Build Transcript Document",
    "Split Into Chunks",
    "Create Embeddings",
    "Build Vector Database",
    "Initialize RAG Chain"
]

# session state initialization
if "step_status" not in st.session_state:
    st.session_state.step_status = {step: "pending" for step in steps}

if "chain" not in st.session_state:
    st.session_state.chain = None

# placeholder so the step list updates instead of duplicating
with pipeline_container:
    step_container = st.empty()


def render_steps():
    with step_container.container():

        st.markdown('<div class="pipeline-box">', unsafe_allow_html=True)

        st.subheader("Pipeline Progress")

        for step in steps:
            status = st.session_state.step_status[step]

            if status == "done":
                st.markdown(f"✅ **{step}**")

            elif status == "running":
                st.markdown(f"🔄 **{step}**")

            else:
                st.markdown(f"⏳ {step}")

        st.markdown("</div>", unsafe_allow_html=True)


st.divider()

with url_container:
    video_url = st.text_input(
        "Paste YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
if video_url:

    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    else:
        video_id = video_url

    with status_container:
        st.info(f"Video ID: {video_id}")

    render_steps()

    if st.session_state.chain is None:

        # Step 1
        st.session_state.step_status["Fetch Transcript"] = "running"
        render_steps()

        transcript_list = get_transcript(video_id)

        st.session_state.step_status["Fetch Transcript"] = "done"
        render_steps()

        if transcript_list:

            # Step 2
            st.session_state.step_status["Build Transcript Document"] = "running"
            render_steps()

            doc = build_document(transcript_list)

            st.session_state.step_status["Build Transcript Document"] = "done"
            render_steps()

            # Step 3
            st.session_state.step_status["Split Into Chunks"] = "running"
            render_steps()

            final_docs = split_with_timestamps(doc)

            st.session_state.step_status["Split Into Chunks"] = "done"
            render_steps()

            # Step 4
            st.session_state.step_status["Create Embeddings"] = "running"
            render_steps()

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.step_status["Create Embeddings"] = "done"
            render_steps()

            # Step 5
            st.session_state.step_status["Build Vector Database"] = "running"
            render_steps()

            vector_store = FAISS.from_documents(final_docs, embeddings)

            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )

            st.session_state.step_status["Build Vector Database"] = "done"
            render_steps()

            # Step 6
            st.session_state.step_status["Initialize RAG Chain"] = "running"
            render_steps()

            chain = build_chain(retriever)

            st.session_state.chain = chain

            st.session_state.step_status["Initialize RAG Chain"] = "done"
            render_steps()

            with status_container:
                st.success("Video ready for questions")


# -------------------------------
# QUESTION SECTION
# -------------------------------

import re

def extract_earliest_timestamp(response_text):
    """
    Extract earliest timestamp from LLM response.
    Expected format: Timestamp: start - end
    """

    timestamps = re.findall(r"Timestamp:\s*([\d\.]+)\s*-\s*([\d\.]+)", response_text)

    if not timestamps:
        return 0

    starts = [float(start) for start, _ in timestamps]

    return int(min(starts))


with qa_container:
    if st.session_state.chain is not None:

        question = st.text_input("Ask a question about the video")

        if question:

            with st.spinner("Generating answer..."):

                response = st.session_state.chain.invoke(question)

            earliest_time = extract_earliest_timestamp(response)

            st.divider()

            col1, col2 = st.columns([1.2,1])

            with col1:

                st.subheader("Answer")

                st.markdown(
                    f'<div class="answer-box">{response}</div>',
                    unsafe_allow_html=True
                )

            with col2:

                st.subheader("Relevant Video Segment")

                embed_url = f"https://www.youtube.com/embed/{video_id}?start={earliest_time}"

                st.components.v1.iframe(embed_url, height=350)

                st.caption(f"Starting at {earliest_time} seconds")