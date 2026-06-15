import streamlit as st
import pymupdf as fitz  # PyMuPDF
import re
import json
import time
import random
import numpy as np
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from fpdf import FPDF
import tempfile
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudyAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: #0F1117; color: #E8EAF0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161B27 !important;
    border-right: 1px solid #1E2533;
}
[data-testid="stSidebar"] .stMarkdown { color: #9BA3B4; }

/* Cards */
.card {
    background: #161B27;
    border: 1px solid #1E2533;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-accent {
    background: linear-gradient(135deg, #1A1F30 0%, #161B27 100%);
    border: 1px solid #2A3550;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #161B27;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1E2533;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #9BA3B4;
    border-radius: 8px;
    font-weight: 500;
    font-size: 14px;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #2A3550 !important;
    color: #7EB8F7 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3B6FE8, #2855C8);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    padding: 0.6rem 1.4rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4F7FF0, #3B6FE8);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(59,111,232,0.3);
}

/* Input */
.stTextInput input, .stTextArea textarea {
    background: #1C2233 !important;
    border: 1px solid #2A3550 !important;
    border-radius: 8px !important;
    color: #E8EAF0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #3B6FE8 !important;
    box-shadow: 0 0 0 2px rgba(59,111,232,0.2) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1C2233;
    border: 2px dashed #2A3550;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover { border-color: #3B6FE8; }

/* Progress & metrics */
.stProgress > div > div { background: #3B6FE8 !important; }
.metric-box {
    background: #1C2233;
    border: 1px solid #2A3550;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-val { font-size: 28px; font-weight: 700; color: #7EB8F7; }
.metric-lbl { font-size: 12px; color: #9BA3B4; margin-top: 2px; }

/* Notes output */
.notes-block {
    background: #1C2233;
    border: 1px solid #2A3550;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    font-size: 14px;
    line-height: 1.7;
}
.chunk-label {
    font-size: 11px;
    font-weight: 600;
    color: #3B6FE8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

/* Chat bubbles */
.msg-user {
    background: #2A3550;
    border-radius: 12px 12px 4px 12px;
    padding: 10px 14px;
    margin: 6px 0 6px 40px;
    font-size: 14px;
    color: #E8EAF0;
}
.msg-ai {
    background: #1C2233;
    border: 1px solid #2A3550;
    border-radius: 12px 12px 12px 4px;
    padding: 10px 14px;
    margin: 6px 40px 6px 0;
    font-size: 14px;
    color: #E8EAF0;
}
.source-pill {
    display: inline-block;
    background: #2A3550;
    color: #7EB8F7;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 99px;
    margin: 4px 2px 0;
    font-family: 'JetBrains Mono', monospace;
}

/* MCQ */
.mcq-card {
    background: #1C2233;
    border: 1px solid #2A3550;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 12px;
}
.mcq-q { font-size: 15px; font-weight: 500; color: #E8EAF0; margin-bottom: 12px; }
.correct-ans { color: #4CAF50; font-weight: 600; font-size: 13px; margin-top: 8px; }
.wrong-ans   { color: #EF5350; font-size: 13px; margin-top: 8px; }
.score-box {
    background: linear-gradient(135deg, #1A2F1A, #1A2333);
    border: 1px solid #2E5E2E;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.score-num { font-size: 48px; font-weight: 700; color: #4CAF50; }
.score-sub { font-size: 14px; color: #9BA3B4; margin-top: 4px; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0F1117; }
::-webkit-scrollbar-thumb { background: #2A3550; border-radius: 99px; }

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
for k, v in {
    "chunks": [], "notes": [], "faiss_index": None, "embeddings_model": None,
    "chunk_embeddings": None, "chat_history": [], "mcq_data": [],
    "mcq_answers": {}, "mcq_submitted": False, "pdf_text": "",
    "processing_done": False, "api_configured": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_text(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp = f.name
    text = ""
    with fitz.open(tmp) as doc:
        for page in doc:
            text += page.get_text()
    os.unlink(tmp)
    return text


def clean_text(text: str) -> str:
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
    return splitter.split_text(text)


def call_gemini(prompt: str, system: str = "") -> str:
    client = st.session_state.client

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": system if system else "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


def summarize_chunk(chunk: str) -> str:
    system = (
        "You are a structured note-taking assistant for exam revision. "
        "Respond ONLY in strict Markdown with headings, bullet points, bold keywords, "
        "code/formula backticks, and a summary table. No prose paragraphs ever."
    )
    prompt = f"""Convert this study material into structured exam notes.

MANDATORY FORMAT:
# 🎯 MAIN TOPIC
## 📌 Subtopic
- **Term**: definition
- Formula: `formula`
## ⚡ Quick Recall Table
| Term | Meaning |
|------|---------|
| X | Y |

RULES: bullets only, bold every keyword, max 12 bullets, end with table.

MATERIAL:
{chunk}"""
    return call_gemini(prompt, system)


def generate_mcq(notes_text: str) -> list:
    prompt = f"""Generate exactly 10 multiple-choice questions from these exam notes.

Return ONLY valid JSON — no markdown, no backticks, no explanation:
[
  {{
    "q": "Question text?",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "A"
  }}
]

NOTES:
{notes_text[:6000]}"""
    raw = call_gemini(prompt)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        # try to extract JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []


def build_faiss_index(chunks: list):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, model, embeddings


def answer_question(question: str) -> str:
    model = st.session_state.embeddings_model
    index = st.session_state.faiss_index
    chunks = st.session_state.chunks

    q_emb = model.encode([question], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k=3)

    context = "\n\n---\n\n".join(
        f"[Chunk {idx+1}]\n{chunks[idx]}"
        for idx in indices[0] if idx < len(chunks)
    )
    source_refs = [f"Chunk {idx+1}" for idx in indices[0] if idx < len(chunks)]

    prompt = f"""Answer the question using ONLY the context provided. Be concise and accurate.
If the answer isn't in the context, say "This topic isn't covered in the uploaded document."

Context:
{context}

Question: {question}

Answer:"""

    answer = call_gemini(prompt)
    return answer, source_refs


def export_pdf(notes: list) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(59, 111, 232)
    pdf.cell(0, 12, "StudyAI — Exam Notes", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(150, 150, 160)
    pdf.cell(0, 6, "Generated by StudyAI", ln=True)
    pdf.ln(4)

    for i, note in enumerate(notes):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(59, 111, 232)
        pdf.cell(0, 8, f"Section {i+1}", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 40)

        for line in note.split("\n"):
            line = line.strip()
            if not line:
                pdf.ln(2)
                continue
            # Clean markdown symbols for PDF
            line = re.sub(r"^#{1,3}\s*", "", line)
            line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            line = re.sub(r"`(.+?)`", r"\1", line)
            line = re.sub(r"^[-•]\s*", "• ", line)
            line = re.sub(r"[^\x00-\x7F]", "", line)  # strip non-ASCII for fpdf
            try:
                pdf.multi_cell(0, 6, line)
            except Exception:
                pass
        pdf.ln(4)

    return pdf.output(dest="S").encode("latin-1")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 StudyAI")
    st.markdown("<p style='color:#9BA3B4;font-size:13px;'>Upload a PDF → get notes, chat, and quiz yourself.</p>", unsafe_allow_html=True)
    st.divider()

    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

if api_key:
    st.session_state.client = Groq(api_key=api_key)
    st.session_state.api_configured = True
    st.success("API key set ✓")

st.markdown("[Get free Groq API key →](https://console.groq.com/keys)")
st.divider()

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded and st.session_state.api_configured:
        if st.button("⚡ Process PDF", use_container_width=True):
            with st.spinner("Extracting text..."):
                raw = extract_text(uploaded)
                cleaned = clean_text(raw)
                st.session_state.pdf_text = cleaned
                chunks = chunk_text(cleaned)
                st.session_state.chunks = chunks

            st.info(f"Split into {len(chunks)} chunks")

            # Notes
            notes = []
            prog = st.progress(0, text="Generating notes...")
            for i, chunk in enumerate(chunks):
                note = summarize_chunk(chunk)
                notes.append(note)
                prog.progress((i + 1) / len(chunks), text=f"Chunk {i+1}/{len(chunks)}")
                time.sleep(0.3)  # rate limit buffer
            st.session_state.notes = notes
            prog.empty()

            # FAISS index
            with st.spinner("Building search index..."):
                index, model, embs = build_faiss_index(chunks)
                st.session_state.faiss_index = index
                st.session_state.embeddings_model = model
                st.session_state.chunk_embeddings = embs

            # MCQ
            with st.spinner("Generating quiz..."):
                combined = "\n\n".join(notes)
                mcqs = generate_mcq(combined)
                st.session_state.mcq_data = mcqs
                st.session_state.mcq_answers = {}
                st.session_state.mcq_submitted = False

            st.session_state.processing_done = True
            st.success("Done! Switch to a tab above.")

    if st.session_state.processing_done:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-box"><div class="metric-val">{len(st.session_state.chunks)}</div><div class="metric-lbl">Chunks</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-box"><div class="metric-val">{len(st.session_state.notes)}</div><div class="metric-lbl">Notes</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-box"><div class="metric-val">{len(st.session_state.mcq_data)}</div><div class="metric-lbl">MCQs</div></div>', unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.api_configured:
    st.markdown("""
    <div class="card-accent" style="text-align:center;padding:3rem;">
        <div style="font-size:48px;margin-bottom:1rem;">🧠</div>
        <h2 style="color:#E8EAF0;margin:0 0 8px;">StudyAI</h2>
        <p style="color:#9BA3B4;font-size:15px;max-width:400px;margin:0 auto;">
            Upload any PDF textbook or notes → get structured exam notes,
            chat with your document, and test yourself with an auto-generated quiz.
        </p>
        <p style="color:#7EB8F7;font-size:13px;margin-top:1.5rem;">← Enter your Gemini API key in the sidebar to get started</p>
    </div>
    """, unsafe_allow_html=True)

elif not st.session_state.processing_done:
    st.markdown("""
    <div class="card" style="text-align:center;padding:2.5rem;">
        <div style="font-size:40px;margin-bottom:1rem;">📄</div>
        <h3 style="color:#E8EAF0;">Upload a PDF to get started</h3>
        <p style="color:#9BA3B4;font-size:14px;">Use the sidebar to upload your PDF and click Process PDF</p>
    </div>
    """, unsafe_allow_html=True)

else:
    tab1, tab2, tab3 = st.tabs(["📝 Exam Notes", "💬 Chat with PDF", "🎯 Quiz"])

    # ── Tab 1: Notes ──────────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 📝 Your Exam Notes")
        with col2:
            if st.button("⬇️ Export PDF", use_container_width=True):
                pdf_bytes = export_pdf(st.session_state.notes)
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name="exam_notes.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        for i, note in enumerate(st.session_state.notes):
            with st.expander(f"Section {i+1}", expanded=(i == 0)):
                st.markdown(note)

    # ── Tab 2: Chat ───────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 💬 Chat with your PDF")
        st.markdown("<p style='color:#9BA3B4;font-size:13px;'>Ask anything about your document — answers come from the actual content.</p>", unsafe_allow_html=True)

        # Chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="msg-user">🙋 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                answer_html = msg["content"].replace("\n", "<br>")
                sources_html = "".join(f'<span class="source-pill">📄 {s}</span>' for s in msg.get("sources", []))
                st.markdown(f'<div class="msg-ai">🤖 {answer_html}<br>{sources_html}</div>', unsafe_allow_html=True)

        # Input
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_input("Ask a question...", placeholder="e.g. What is a deadlock? Explain with example.")
            submitted = st.form_submit_button("Send →", use_container_width=True)

        if submitted and question.strip():
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Searching your document..."):
                answer, sources = answer_question(question)
            st.session_state.chat_history.append({"role": "ai", "content": answer, "sources": sources})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear chat", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()

    # ── Tab 3: Quiz ───────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🎯 Auto-Generated Quiz")

        if not st.session_state.mcq_data:
            st.warning("No quiz questions generated. Try re-processing the PDF.")
        else:
            if not st.session_state.mcq_submitted:
                st.markdown(f"<p style='color:#9BA3B4;font-size:13px;'>Answer all {len(st.session_state.mcq_data)} questions, then submit.</p>", unsafe_allow_html=True)

                for i, q in enumerate(st.session_state.mcq_data):
                    st.markdown(f'<div class="mcq-card"><div class="mcq-q">Q{i+1}. {q["q"]}</div></div>', unsafe_allow_html=True)
                    choice = st.radio(
                        f"q{i}",
                        options=q["options"],
                        key=f"mcq_{i}",
                        label_visibility="collapsed",
                    )
                    if choice:
                        st.session_state.mcq_answers[i] = choice[0]  # "A", "B" etc.

                st.divider()
                if st.button("✅ Submit Quiz", use_container_width=True):
                    st.session_state.mcq_submitted = True
                    st.rerun()

            else:
                # Results
                correct = sum(
                    1 for i, q in enumerate(st.session_state.mcq_data)
                    if st.session_state.mcq_answers.get(i) == q["answer"]
                )
                total = len(st.session_state.mcq_data)
                pct = int(correct / total * 100)

                grade = "🏆 Excellent!" if pct >= 80 else "👍 Good effort!" if pct >= 50 else "📖 Keep revising!"
                st.markdown(f"""
                <div class="score-box">
                    <div class="score-num">{correct}/{total}</div>
                    <div class="score-sub">{pct}% · {grade}</div>
                </div>
                """, unsafe_allow_html=True)
                st.divider()

                for i, q in enumerate(st.session_state.mcq_data):
                    user_ans = st.session_state.mcq_answers.get(i, "—")
                    correct_ans = q["answer"]
                    is_correct = user_ans == correct_ans

                    icon = "✅" if is_correct else "❌"
                    border_color = "#2E5E2E" if is_correct else "#5E2E2E"
                    correct_opt = next((o for o in q["options"] if o.startswith(correct_ans)), correct_ans)

                    st.markdown(f"""
                    <div class="mcq-card" style="border-color:{border_color}">
                        <div class="mcq-q">{icon} Q{i+1}. {q["q"]}</div>
                        <div style="font-size:13px;color:#9BA3B4;">Your answer: {user_ans}</div>
                        {"" if is_correct else f'<div class="correct-ans">✓ Correct: {correct_opt}</div>'}
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()
                if st.button("🔄 Retake Quiz", use_container_width=True):
                    st.session_state.mcq_answers = {}
                    st.session_state.mcq_submitted = False
                    st.rerun()
