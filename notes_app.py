import streamlit as st
import fitz
import re
import json
import time
import numpy as np
import faiss
import tempfile
import os

from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from fpdf import FPDF
from io import BytesIO

# ───────────────────────── CONFIG ─────────────────────────
st.set_page_config(page_title="StudyAI Pro", page_icon="🧠", layout="wide")

MAX_CHUNKS_TO_PROCESS = 5        # 🔥 prevents API overload
API_SLEEP = 1.2                  # 🔥 prevents burst rate limit


# ───────────────────────── SESSION STATE ─────────────────────────
defaults = {
    "chunks": [],
    "notes": [],
    "faiss_index": None,
    "embeddings_model": None,
    "chat_history": [],
    "mcq_data": [],
    "mcq_answers": {},
    "mcq_submitted": False,
    "api_key": "",
    "ready": False,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ───────────────────────── CACHE MODEL ─────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ───────────────────────── GROQ CLIENT ─────────────────────────
def get_client():
    return Groq(api_key=st.session_state.api_key)


def llm(prompt, system=""):
    client = get_client()
    messages = []

    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2,
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"


# ───────────────────────── PDF ─────────────────────────
def extract_text(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file.read())
        path = f.name

    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()

    os.unlink(path)
    return text


def clean_text(t):
    t = "\n".join(line.strip() for line in t.splitlines())
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# ───────────────────────── NOTES ─────────────────────────
def summarize(chunk):
    prompt = f"""
Convert into structured exam notes:

- headings
- bullet points
- bold keywords
- table if needed

TEXT:
{chunk[:6000]}
"""
    return llm(prompt).strip()


# ───────────────────────── MCQ (CACHED) ─────────────────────────
@st.cache_data(show_spinner=False)
def generate_mcq(text):
    prompt = f"""
Generate 10 MCQs.

Return ONLY JSON:
[
  {{
    "q": "...",
    "options": ["A) ...","B) ...","C) ...","D) ..."],
    "answer": "A"
  }}
]

TEXT:
{text[:6000]}
"""

    raw = llm(prompt)
    raw = re.sub(r"```.*?```", "", raw, flags=re.S).strip()

    try:
        return json.loads(raw)
    except:
        match = re.search(r"\[.*\]", raw, re.S)
        return json.loads(match.group()) if match else []


# ───────────────────────── FAISS ─────────────────────────
@st.cache_resource
def build_index(chunks):
    model = load_model()
    emb = model.encode(chunks)
    emb = np.array(emb, dtype="float32")

    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    return index, model


# ───────────────────────── QA ─────────────────────────
def answer_question(q):
    model = st.session_state.embeddings_model
    index = st.session_state.faiss_index
    chunks = st.session_state.chunks

    q_emb = model.encode([q])
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)

    _, idxs = index.search(q_emb, 3)

    context = "\n\n".join(chunks[i] for i in idxs[0])

    prompt = f"""
Answer ONLY using context.

Context:
{context}

Question: {q}
"""

    ans = llm(prompt)
    sources = [f"Chunk {i+1}" for i in idxs[0]]
    return ans, sources


# ───────────────────────── PDF EXPORT ─────────────────────────
def export_pdf(notes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "StudyAI Notes", ln=True)
    pdf.ln(5)

    def clean_line(text):
        # remove non-ascii + collapse long tokens
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"\s+", " ", text)

        # HARD FIX: break extremely long words
        return " ".join([w if len(w) < 80 else w[:80] + "..." for w in text.split(" ")])

    for i, n in enumerate(notes):
        pdf.set_font("Arial", style="B", size=11)
        pdf.cell(200, 10, f"Section {i+1}", ln=True)

        pdf.set_font("Arial", size=10)

        for line in n.split("\n"):
            line = clean_line(line)

            if not line.strip():
                continue

            try:
                pdf.multi_cell(0, 6, line)
            except:
                # LAST RESORT: skip bad line instead of crashing
                continue

        pdf.ln(3)

    buffer = BytesIO()
    buffer.write(pdf.output(dest="S").encode("latin-1"))
    buffer.seek(0)

    return buffer
# ───────────────────────── SIDEBAR ─────────────────────────
with st.sidebar:
    st.title("🧠 StudyAI Pro")

    key = st.text_input("Groq API Key", type="password")
    if key:
        st.session_state.api_key = key

    file = st.file_uploader("Upload PDF")

    if file and st.button("Process"):
        raw = extract_text(file)
        clean = clean_text(raw)

        chunks = chunk_text(clean)
        st.session_state.chunks = chunks[:MAX_CHUNKS_TO_PROCESS]  # 🔥 LIMIT

        # ── NOTES (SAFE LOOP) ──
        notes = []
        for i, c in enumerate(st.session_state.chunks):
            notes.append(summarize(c))
            time.sleep(API_SLEEP)   # 🔥 prevents rate limit

        st.session_state.notes = notes

        # ── INDEX ──
        index, model = build_index(st.session_state.chunks)
        st.session_state.faiss_index = index
        st.session_state.embeddings_model = model

        # ── MCQ ──
        combined = "\n".join(notes)
        st.session_state.mcq_data = generate_mcq(combined)

        st.session_state.ready = True
        st.success("Ready!")


# ───────────────────────── MAIN UI ─────────────────────────
if not st.session_state.ready:
    st.info("Upload PDF + API key")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Notes", "Chat", "Quiz"])


# ───────── NOTES ─────────
with tab1:
    st.markdown("### 📝 Your Exam Notes")

    # Generate PDF once on click
    if st.button("📄 Generate PDF"):
        st.session_state.pdf_buffer = export_pdf(st.session_state.notes)
        st.success("PDF ready for download!")

    # Persistent download button (IMPORTANT FIX)
    if "pdf_buffer" in st.session_state:
        st.download_button(
            label="⬇️ Download PDF",
            data=st.session_state.pdf_buffer,
            file_name="notes.pdf",
            mime="application/pdf"
        )

    # Notes display
    for i, n in enumerate(st.session_state.notes):
        st.markdown(f"### Section {i+1}")
        st.markdown(n)


# ───────── CHAT ─────────
with tab2:
    for m in st.session_state.chat_history[-6:]:
        st.write(m)

    q = st.text_input("Ask")
    if q:
        ans, src = answer_question(q)
        st.session_state.chat_history.append({"q": q, "a": ans})

        st.write(ans)
        st.caption(", ".join(src))


# ───────── QUIZ ─────────
with tab3:
    if not st.session_state.mcq_data:
        st.warning("No MCQs generated")
    else:
        for i, q in enumerate(st.session_state.mcq_data):
            st.write(q["q"])
            ans = st.radio("", q["options"], key=i)
            st.session_state.mcq_answers[i] = ans[0]

        if st.button("Submit"):
            correct = sum(
                1 for i, q in enumerate(st.session_state.mcq_data)
                if st.session_state.mcq_answers.get(i) == q["answer"]
            )
            st.success(f"Score: {correct}/{len(st.session_state.mcq_data)}")
