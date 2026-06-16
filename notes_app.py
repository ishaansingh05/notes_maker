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
st.set_page_config(page_title="Exam Guide", page_icon="🧠", layout="wide")

MAX_CHUNKS_TO_PROCESS = 5
API_SLEEP = 1.2

# ───────────────────────── CUSTOM CSS ─────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ── Sidebar title ── */
[data-testid="stSidebar"] h1 {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1.5rem !important;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
    transition: border 0.2s;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.2) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05) !important;
    border: 2px dashed rgba(167,139,250,0.4) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #a78bfa !important;
}

/* ── Primary Button ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.6) !important;
}

/* ── Download Button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #0d9488) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(5,150,105,0.4) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(5,150,105,0.6) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    box-shadow: 0 2px 10px rgba(124,58,237,0.4) !important;
}

/* ── Tab content ── */
.stTabs [data-baseweb="tab-panel"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 0 12px 12px 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    padding: 1.5rem !important;
    margin-top: -1px !important;
}

/* ── Markdown text ── */
.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: #cbd5e1 !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #e2e8f0 !important;
}
.stMarkdown strong {
    color: #a78bfa !important;
}
.stMarkdown h3 {
    border-left: 3px solid #7c3aed;
    padding-left: 0.75rem;
    margin-top: 1.5rem !important;
}

/* ── Radio buttons (quiz) ── */
.stRadio label {
    color: #cbd5e1 !important;
    font-size: 0.95rem !important;
}
.stRadio [data-testid="stMarkdownContainer"] p {
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
}

/* ── Text input labels ── */
label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ── Success / Info / Warning ── */
.stSuccess {
    background: rgba(5,150,105,0.15) !important;
    border: 1px solid rgba(5,150,105,0.3) !important;
    border-radius: 10px !important;
    color: #6ee7b7 !important;
}
.stInfo {
    background: rgba(79,70,229,0.15) !important;
    border: 1px solid rgba(79,70,229,0.3) !important;
    border-radius: 10px !important;
    color: #a5b4fc !important;
}
.stWarning {
    background: rgba(217,119,6,0.15) !important;
    border: 1px solid rgba(217,119,6,0.3) !important;
    border-radius: 10px !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #7c3aed !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb {
    background: rgba(124,58,237,0.4);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,0.7); }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Section card ── */
.section-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* ── Score badge ── */
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    font-size: 1.4rem;
    font-weight: 700;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    box-shadow: 0 4px 20px rgba(124,58,237,0.5);
    margin-top: 1rem;
}

/* ── Chat message ── */
.chat-q {
    background: rgba(124,58,237,0.15);
    border-left: 3px solid #7c3aed;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    color: #c4b5fd !important;
    font-weight: 500;
}
.chat-a {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #60a5fa;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────── SESSION STATE ─────────────────────────
defaults = {
    "chunks": [], "notes": [], "faiss_index": None,
    "embeddings_model": None, "chat_history": [],
    "mcq_data": {}, "mcq_answers": {}, "mcq_submitted": False,
    "api_key": "", "ready": False,
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
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

# ───────────────────────── MCQ ─────────────────────────
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

    def clean(text):
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    for i, n in enumerate(notes):
        pdf.set_font("Arial", style="B", size=11)
        pdf.cell(200, 10, f"Section {i+1}", ln=True)
        pdf.set_font("Arial", size=10)
        for line in n.split("\n"):
            line = clean(line).strip()
            if not line:
                continue
            try:
                pdf.multi_cell(0, 6, line)
            except:
                continue
        pdf.ln(3)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# ───────────────────────── SIDEBAR ─────────────────────────
with st.sidebar:
    st.markdown("# 🧠 EXAM GUIDE")
    st.markdown("---")

    st.markdown("#### 🔑 API Key")
    key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if key:
        st.session_state.api_key = key

    st.markdown("#### 📂 Upload Material")
    file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    st.markdown("")
    if file and st.button("⚡ Process Document"):
        with st.spinner("Reading PDF..."):
            raw = extract_text(file)
            clean = clean_text(raw)
            chunks = chunk_text(clean)
            st.session_state.chunks = chunks[:MAX_CHUNKS_TO_PROCESS]

        with st.spinner("✍️ Generating notes..."):
            notes = []
            for i, c in enumerate(st.session_state.chunks):
                notes.append(summarize(c))
                time.sleep(API_SLEEP)
            st.session_state.notes = notes

        with st.spinner("🔍 Building search index..."):
            index, model = build_index(st.session_state.chunks)
            st.session_state.faiss_index = index
            st.session_state.embeddings_model = model

        with st.spinner("🧩 Generating quiz..."):
            combined = "\n".join(notes)
            st.session_state.mcq_data = generate_mcq(combined)

        st.session_state.ready = True
        st.success("✅ Ready! Switch to a tab above.")

    st.markdown("---")
    st.markdown(
        "<div style='color:#475569;font-size:0.78rem;text-align:center'>"
        "Powered by Groq · FAISS · LangChain"
        "</div>",
        unsafe_allow_html=True
    )

# ───────────────────────── HERO ─────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 Exam Guide</h1>
    <p>Upload your study material · Get smart notes · Chat with your PDF · Take a quiz</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.ready:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.info("👈 Add your Groq API key and upload a PDF in the sidebar to get started.")
    st.stop()

# ───────────────────────── TABS ─────────────────────────
tab1, tab2, tab3 = st.tabs(["📝  Notes", "💬  Chat", "🧩  Quiz"])

# ─────────── NOTES TAB ───────────
with tab1:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown("### 📝 Your Exam Notes")
    with col_b:
        if st.button("📄 Generate PDF"):
            st.session_state.pdf_buffer = export_pdf(st.session_state.notes)
            st.success("PDF ready!")

    if "pdf_buffer" in st.session_state:
        st.download_button(
            label="⬇️ Download PDF Notes",
            data=st.session_state.pdf_buffer,
            file_name="exam_notes.pdf",
            mime="application/pdf"
        )

    st.markdown("---")
    for i, n in enumerate(st.session_state.notes):
        st.markdown(
            f'<div class="section-card"><strong style="color:#a78bfa">Section {i+1}</strong></div>',
            unsafe_allow_html=True
        )
        st.markdown(n)

# ─────────── CHAT TAB ───────────
with tab2:
    st.markdown("### 💬 Ask Anything About Your Material")

    # Display history
    for m in st.session_state.chat_history[-6:]:
        st.markdown(
            f'<div class="chat-q">🙋 {m["q"]}</div>'
            f'<div class="chat-a">🤖 {m["a"]}</div>',
            unsafe_allow_html=True
        )

    q = st.text_input(
        "Your question",
        placeholder="e.g. What is the main concept of chapter 2?",
        label_visibility="collapsed"
    )
    if q:
        with st.spinner("Thinking..."):
            ans, src = answer_question(q)
        st.session_state.chat_history.append({"q": q, "a": ans})

        st.markdown(
            f'<div class="chat-q">🙋 {q}</div>'
            f'<div class="chat-a">🤖 {ans}</div>',
            unsafe_allow_html=True
        )
        st.caption("📎 Sources: " + ", ".join(src))

# ─────────── QUIZ TAB ───────────
with tab3:
    st.markdown("### 🧩 Test Your Knowledge")

    if not st.session_state.mcq_data:
        st.warning("No quiz questions generated yet. Process a PDF first.")
    else:
        if not st.session_state.mcq_submitted:
            with st.form("quiz_form"):
                for i, q in enumerate(st.session_state.mcq_data):
                    st.markdown(
                        f'<div class="section-card">'
                        f'<strong style="color:#a78bfa">Q{i+1}.</strong> '
                        f'<span style="color:#e2e8f0">{q["q"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    ans = st.radio("", q["options"], key=f"q_{i}", label_visibility="collapsed")
                    st.session_state.mcq_answers[i] = ans[0] if ans else ""
                    st.markdown("")

                submitted = st.form_submit_button("🎯 Submit Quiz", use_container_width=True)
                if submitted:
                    st.session_state.mcq_submitted = True
                    st.rerun()
        else:
            correct = sum(
                1 for i, q in enumerate(st.session_state.mcq_data)
                if st.session_state.mcq_answers.get(i) == q["answer"]
            )
            total = len(st.session_state.mcq_data)
            pct = int(correct / total * 100)

            grade = "🏆 Excellent!" if pct >= 80 else "👍 Good job!" if pct >= 60 else "📚 Keep studying!"

            st.markdown(
                f'<div style="text-align:center;padding:2rem">'
                f'<div class="score-badge">{correct}/{total} · {pct}%</div>'
                f'<p style="color:#94a3b8;margin-top:1rem;font-size:1.1rem">{grade}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown("---")
            st.markdown("#### 📋 Answer Review")
            for i, q in enumerate(st.session_state.mcq_data):
                user_ans = st.session_state.mcq_answers.get(i, "")
                correct_ans = q["answer"]
                is_correct = user_ans == correct_ans
                icon = "✅" if is_correct else "❌"
                color = "#6ee7b7" if is_correct else "#fca5a5"
                st.markdown(
                    f'<div class="section-card">'
                    f'{icon} <strong style="color:{color}">Q{i+1}.</strong> '
                    f'<span style="color:#e2e8f0">{q["q"]}</span><br>'
                    f'<span style="color:#94a3b8;font-size:0.9rem">Your answer: {user_ans} · '
                    f'Correct: {correct_ans}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            if st.button("🔄 Retake Quiz"):
                st.session_state.mcq_submitted = False
                st.session_state.mcq_answers = {}
                st.rerun()
