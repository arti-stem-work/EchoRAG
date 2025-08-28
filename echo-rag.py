import os
import io
import tempfile
import sys
from typing import List, Dict, Any, Tuple


# --- Workaround: force modern sqlite on hosts like Streamlit Cloud ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except Exception:
    pass


import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import PyPDF2


try:
    import whisper
    HAS_WHISPER = True
except Exception:
    HAS_WHISPER = False


import requests
from gtts import gTTS


# ------------------------------ Configuration ------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # compact, fast embedding
CHROMA_PERSIST_DIR = "./chroma_db" # change to None for in-memory
TOP_K = 5


HF_API_URL = "https://api-inference.huggingface.co/models/" # + model
HF_API_KEY = os.getenv("HF_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ------------------------------ Utilities ------------------------------
@st.cache_resource
def init_embedding_model(model_name: str = EMBEDDING_MODEL):
    return SentenceTransformer(model_name)

@st.cache_resource
def init_chroma(persist_directory: str = CHROMA_PERSIST_DIR):
    # Auto-detect environment: Streamlit Cloud often has old sqlite → fallback to in-memory
    if os.getenv("STREAMLIT_RUNTIME") == "1":
        persist_directory = None
    chroma_settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
    client = chromadb.Client(chroma_settings)
    return client
    
# Simple chunking: split on whitespace up to approx chunk_size words with overlap
def chunk_text(text: str, chunk_size_words: int = 250, overlap_words: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size_words]
        chunks.append(" ".join(chunk))
        i += chunk_size_words - overlap_words
    return chunks

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text)

# Whisper transcription helper (local)
def transcribe_audio_whisper_local(audio_path: str, model_name: str = "small") -> str:
    if not HAS_WHISPER:
        raise RuntimeError("Whisper is not installed in this environment")
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result.get("text", "")

# HF Whisper inference helper
def transcribe_audio_whisper_hf(audio_bytes: bytes, hf_api_key: str, model: str = "openai/whisper-small") -> str:
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    files = {"file": ("audio.wav", audio_bytes)}
    url = f"https://api-inference.huggingface.co/models/{model}"
    resp = requests.post(url, headers=headers, files=files, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HF Whisper inference failed: {resp.status_code} {resp.text}")
    data = resp.json()
    # HF whisper returns text in different formats depending on model; try common keys
    if isinstance(data, dict) and "text" in data:
        return data["text"]
    if isinstance(data, dict) and "transcription" in data:
        return data["transcription"]
    # fallback
    return str(data)

# Create embeddings via sentence-transformers; returns list of vectors
def embed_texts(encoder: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    embeddings = encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()

# Initialize Chroma collection
def get_or_create_collection(client: chromadb.Client, name: str = "multimodal_docs"):
    try:
        collection = client.get_collection(name)
    except Exception:
        # create embedding function wrapper so Chroma can accept our vectors directly
        collection = client.create_collection(name)
    return collection

# Store texts in Chroma
def store_chunks_in_chroma(collection, ids: List[str], metadatas: List[Dict[str, Any]], texts: List[str], embeddings: List[List[float]]):
    collection.add(ids=ids, metadatas=metadatas, documents=texts, embeddings=embeddings)

# Query Chroma for top_k similar chunks
def retrieve_from_chroma(collection, query_embedding: List[float], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=['documents','metadatas','distances'])
    # results come as dict with lists
    docs = []
    for i in range(len(results['documents'][0])):
        docs.append({
            'document': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        })
    return docs

# Call HF generative model via Inference API
def call_hf_generation(hf_api_key: str, model: str, prompt: str, max_tokens: int = 256) -> str:
    headers = {"Authorization": f"Bearer {hf_api_key}", "Content-Type": "application/json"}
    url = HF_API_URL + model
    payload = {"inputs": prompt, "options": {"wait_for_model": True, "use_cache": False}}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HF generation failed: {resp.status_code} {resp.text}")
    data = resp.json()
    # HF sometimes returns list of dicts
    if isinstance(data, list) and len(data) > 0 and 'generated_text' in data[0]:
        return data[0]['generated_text']
    if isinstance(data, dict) and 'generated_text' in data:
        return data['generated_text']
    # fallback: stringify
    return str(data)

# Optional: OpenAI completion wrapper (if OPENAI_API_KEY present)
def call_openai_completion(openai_api_key: str, prompt: str, model: str = "gpt-3.5-turbo") -> str:
    try:
        import openai
    except Exception:
        raise RuntimeError("openai package not installed")
    openai.api_key = openai_api_key
    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        # chat completion
        resp = openai.ChatCompletion.create(model=model, messages=[{"role":"user","content":prompt}], max_tokens=512)
        return resp['choices'][0]['message']['content']
    else:
        resp = openai.Completion.create(model=model, prompt=prompt, max_tokens=512)
        return resp['choices'][0]['text']

# TTS via gTTS (returns bytes)
def synthesize_tts_gtts(text: str, lang: str = 'en') -> bytes:
    tts = gTTS(text)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()

# ------------------------------ Streamlit App ------------------------------

st.set_page_config(page_title="Multimodal RAG — Streamlit Demo", layout="wide")
st.title("Multimodal Retrieval-Augmented Generation (RAG) — Streamlit Demo")
st.caption("Enterprise-minded prototype: text + audio ingestion → vector store → RAG → TTS audio response")

# Sidebar: configuration
with st.sidebar:
    st.header("Configuration")
    persist = st.checkbox("Persist Chroma to disk (CHROMA_PERSIST_DIR)", value=True)
    model_choice = st.selectbox("Generative model backend", options=["huggingface_gpt-j", "openai_gpt-3.5 (requires OPENAI_API_KEY)"], index=0)
    whisper_backend = st.selectbox("Audio transcription backend", options=["local_whisper (if installed)", "huggingface_whisper_api"], index=1)
    top_k = st.slider("Retrieval top_k", min_value=1, max_value=10, value=TOP_K)

# Initialize resources
encoder = init_embedding_model()
client = init_chroma(CHROMA_PERSIST_DIR if persist else None)
collection = get_or_create_collection(client, name="multimodal_docs")

# Main UI
st.header("1) Upload documents and audio")
uploaded_files = st.file_uploader("Upload text documents (txt / pdf) or audio files (wav/mp3). You may upload multiple.", accept_multiple_files=True)

if uploaded_files:
    ingest_button = st.button("Ingest uploaded files into vector store")
    if ingest_button:
        progress = st.progress(0)
        total = len(uploaded_files)
        for idx, up in enumerate(uploaded_files):
            name = up.name
            st.write(f"Processing: {name}")
            if name.lower().endswith('.pdf'):
                raw = up.read()
                text = extract_text_from_pdf(raw)
                chunks = chunk_text(text)
                embeddings = embed_texts(encoder, chunks)
                ids = [f"pdf_{name}_{i}" for i in range(len(chunks))]
                metadatas = [{"source": name, "type": "pdf", "chunk_index": i} for i in range(len(chunks))]
                store_chunks_in_chroma(collection, ids, metadatas, chunks, embeddings)
            elif name.lower().endswith('.txt'):
                raw = up.read().decode('utf-8')
                chunks = chunk_text(raw)
                embeddings = embed_texts(encoder, chunks)
                ids = [f"txt_{name}_{i}" for i in range(len(chunks))]
                metadatas = [{"source": name, "type": "txt", "chunk_index": i} for i in range(len(chunks))]
                store_chunks_in_chroma(collection, ids, metadatas, chunks, embeddings)
            elif name.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                # save to temp file
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1])
                tmpf.write(up.read())
                tmpf.flush()
                tmpf.close()
                transcript = ""
                try:
                    if whisper_backend.startswith('local') and HAS_WHISPER:
                        transcript = transcribe_audio_whisper_local(tmpf.name)
                    else:
                        if not HF_API_KEY:
                            st.error("HF_API_KEY required for HuggingFace Whisper inference")
                        else:
                            with open(tmpf.name, 'rb') as f:
                                transcript = transcribe_audio_whisper_hf(f.read(), HF_API_KEY)
                except Exception as e:
                    st.error(f"Audio transcription failed: {e}")
                    transcript = ""
                if transcript:
                    chunks = chunk_text(transcript)
                    embeddings = embed_texts(encoder, chunks)
                    ids = [f"audio_{name}_{i}" for i in range(len(chunks))]
                    metadatas = [{"source": name, "type": "audio", "chunk_index": i} for i in range(len(chunks))]
                    store_chunks_in_chroma(collection, ids, metadatas, chunks, embeddings)
                else:
                    st.warning(f"No transcript produced for {name}")
            else:
                st.warning(f"Unsupported file type: {name}")
            progress.progress(int((idx + 1) / total * 100))
        st.success("Ingestion completed.")
        # optionally persist
        if persist:
            try:
                client.persist()
                st.info("Chroma DB persisted to disk.")
            except Exception:
                st.warning("Chroma persist not available in this environment.")

st.header("2) Query the corpus / Ask a question")
query_text = st.text_input("Enter query text:", placeholder="Ask about the uploaded documents or audio...")
if st.button("Run query") and query_text:
    # Embed query
    q_emb = encoder.encode([query_text], convert_to_numpy=True)[0].tolist()
    docs = retrieve_from_chroma(collection, q_emb, top_k=top_k)
    st.subheader("Retrieved context (top results)")
    context_texts = []
    for i, d in enumerate(docs):
        st.markdown(f"**Result {i+1}** — source: {d['metadata'].get('source')} (distance: {d['distance']:.4f})")
        st.write(d['document'])
        context_texts.append(d['document'])

    # Build prompt for LLM
    assembled_context = "\n---\n".join(context_texts)
    prompt = (
        "You are an enterprise AI assistant. Use the context below to answer the user's question. "
        "If the answer is not contained in the context, say 'I don't know'.\n\n"
        f"CONTEXT:\n{assembled_context}\n\nQUESTION:\n{query_text}\n\nAnswer concisely."
    )

    # Call the generative model
    response_text = ""
    try:
        if model_choice.startswith('huggingface'):
            if not HF_API_KEY:
                st.error("HF_API_KEY is required for Hugging Face generation")
                raise RuntimeError("Missing HF_API_KEY")
            # example: use 'EleutherAI/gpt-j-6B' or a smaller instruct-tuned model
            hf_model = 'EleutherAI/gpt-j-6B'
            response_text = call_hf_generation(HF_API_KEY, hf_model, prompt)
        else:
            if not OPENAI_API_KEY:
                st.error("OPENAI_API_KEY required for OpenAI backend")
            else:
                response_text = call_openai_completion(OPENAI_API_KEY, prompt, model='gpt-3.5-turbo')
    except Exception as e:
        st.error(f"Generative model call failed: {e}")
        response_text = ""

    if response_text:
        st.subheader("LLM Response")
        st.write(response_text)

        # TTS: synthesize and stream audio
        tts_bytes = synthesize_tts_gtts(response_text)
        st.audio(tts_bytes, format='audio/mp3')
        st.download_button("Download response text", response_text, file_name='response.txt')

