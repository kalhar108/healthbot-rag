import os, asyncio, textwrap, time
import streamlit as st
import nest_asyncio
from groq import Groq

nest_asyncio.apply()

# ========== CONFIG ==========
st.set_page_config(page_title="HealthBot (RAG)", page_icon="🩺", layout="wide")

# Sidebar: Secrets / config
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "")
PINECONE_API_KEY  = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY", "")
PINECONE_HOST     = st.secrets.get("PINECONE_HOST") or os.getenv("PINECONE_HOST", "")
PINECONE_INDEX    = st.secrets.get("PINECONE_INDEX_NAME") or os.getenv("PINECONE_INDEX_NAME", "medicalbot")

if not (GROQ_API_KEY and PINECONE_API_KEY and PINECONE_HOST and PINECONE_INDEX):
    st.warning("Missing keys/host/index. Please add GROQ_API_KEY, PINECONE_API_KEY, PINECONE_HOST, and PINECONE_INDEX_NAME in Streamlit secrets.", icon="⚠️")

# Imports that rely on installed deps (ensure requirements.txt)
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import wikipedia

# Embedder & Reranker
EMBED_MODEL_NAME = st.secrets.get("EMBED_MODEL_NAME") or os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
_embedder = SentenceTransformer(EMBED_MODEL_NAME)
_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    embs = _embedder.encode(texts, normalize_embeddings=True)
    return np.array(embs).tolist()

# Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

# ========== RETRIEVAL ==========

async def query_pinecone(question: str, top_k: int = 8, namespace: str = ""):
    vec = embed(question)[0]
    res = index.query(vector=vec, top_k=top_k, include_metadata=True, namespace=namespace or None)
    hits = []
    for m in res.get("matches", []):
        hits.append({
            "id": m["id"],
            "score": m["score"],
            "text": (m.get("metadata") or {}).get("text") or (m.get("metadata") or {}).get("chunk") or "",
            "source": (m.get("metadata") or {}).get("source") or (m.get("metadata") or {}).get("file") or "",
            "meta": m.get("metadata") or {},
        })
    return hits

async def query_wikipedia(question: str, max_chars: int = 2000):
    try:
        titles = wikipedia.search(question)[:3]
        chunks = []
        for t in titles:
            try:
                page = wikipedia.page(t, auto_suggest=False)
                content = page.content[:max_chars]
                chunks.append({
                    "id": f"wiki::{t}",
                    "score": 0.0,
                    "text": content,
                    "source": page.url,
                    "meta": {"title": t, "provider": "wikipedia"}
                })
            except Exception:
                pass
        return chunks
    except Exception:
        return []

def rerank(question: str, docs: list, top_k: int = 6):
    if not docs:
        return []
    pairs = [(question, d["text"][:1000]) for d in docs]
    scores = _reranker.predict(pairs)
    for d, s in zip(docs, scores):
        d["rerank_score"] = float(s)
    return sorted(docs, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]

# ========== PROMPTING ==========

def build_system_prompt(tool: str) -> str:
    base = (
        "You are HealthBot, a careful medical information assistant. "
        "You provide general, educational information only — never medical advice. "
        "Always include citations to the retrieved sources. "
        "Highlight any red flags and recommend seeking professional care when appropriate.\n"
    )
    tool_text = {
        "disease_guide": "Focus on definition, key symptoms, common differentials, red flags, and when to seek care.",
        "drug_lookup": "Summarize indications, contraindications, side effects, and interactions. Do not provide dosing.",
        "triage": "Give a risk tier (low/moderate/high) with prudent next steps and red flags.",
        "labs_explainer": "Explain what a lab generally measures and common causes of high/low values.",
        "lifestyle_coach": "Offer general lifestyle guidance aligned with reputable guidelines.",
    }.get(tool, "")
    safety = "Never prescribe medications or give individualized advice. Use disclaimers. Be concise and structured."
    return base + tool_text + "\n" + safety

def format_context(docs):
    blks = []
    for d in docs:
        src = d.get("source") or d.get("meta", {}).get("title", "")
        blks.append(f"[Source: {src}]\n{d['text']}\n")
    return "\n---\n".join(blks)

def build_user_prompt(question: str, tool: str, docs):
    context = format_context(docs)
    return f"""
User question (tool={tool}):
{question}

Use the retrieved context. If something is unclear or missing, state the limitation explicitly.
Provide bullet points and short paragraphs. End with a compact "Sources" list.

Context:
{context}
""".strip()

# ========== GROQ MODEL ==========

def get_groq_client():
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY. Please add it to Streamlit secrets.")
    return Groq(api_key=GROQ_API_KEY)

def answer_with_groq(question: str, tool: str, docs, model: str = "llama-3.3-70b-versatile") -> str:
    client = get_groq_client()
    system_prompt = build_system_prompt(tool)
    user_prompt = build_user_prompt(question, tool, docs)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=1200,
        temperature=0.2,
    )
    return response.choices[0].message.content if response and response.choices else ""

# ========== PIPELINE ==========

async def retrieve_and_generate(question: str, tool: str, top_k: int = 8, namespace: str = ""):
    pine_task = asyncio.create_task(query_pinecone(question, top_k=top_k, namespace=namespace))
    wiki_task = asyncio.create_task(query_wikipedia(question))
    pine_hits, wiki_hits = await asyncio.gather(pine_task, wiki_task)
    all_docs = (pine_hits or []) + (wiki_hits or [])
    picked = rerank(question, all_docs, top_k=6) if all_docs else []
    answer = answer_with_groq(question, tool, picked)
    return {"answer": answer, "docs": picked}

# ========== UI ==========

st.title("🩺 HealthBot (RAG) — Groq + Pinecone")
st.caption("Educational medical information only • Not medical advice")

with st.sidebar:
    st.header("Settings")
    tool = st.selectbox(
        "Choose a tool",
        ["disease_guide", "drug_lookup", "triage", "labs_explainer", "lifestyle_coach"],
        index=0,
        help="Each tool adjusts prompts and retrieval strategy."
    )
    namespace = st.text_input("Pinecone namespace (optional)", value="")
    top_k = st.slider("Top-K Retrieval", 4, 16, 8, 1)
    st.markdown("---")
    st.write("**Keys loaded:**",
             "- Groq ✅" if GROQ_API_KEY else "- Groq ❌")
    st.write("**Pinecone:**", "✅" if (PINECONE_API_KEY and PINECONE_HOST) else "❌")

query = st.text_input("Ask a health question", placeholder="e.g., What red flags for chest pain warrant urgent care?")
go = st.button("Run")

if go and query.strip():
    with st.spinner("Retrieving & generating..."):
        out = asyncio.run(retrieve_and_generate(query.strip(), tool=tool, top_k=top_k, namespace=namespace))
    st.subheader("Answer")
    st.write(out["answer"] or "_No answer produced_")

    st.subheader("Retrieved Context")
    for i, d in enumerate(out["docs"], 1):
        with st.expander(f"{i}. {d.get('source','<no source>')} | score={round(d.get('rerank_score',0),3)}"):
            st.write(d.get("text","")[:1500])
            if d.get("source"):
                st.write(f"[Open Source]({d['source']})")
else:
    st.info("Enter a question and click **Run**.", icon="💡")

st.markdown("---")
st.caption("Built with ❤️ using Groq + Pinecone • Informational only, not medical advice.")
