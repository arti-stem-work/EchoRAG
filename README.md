Multimodal RAG Streamlit App 
An end-to-end Streamlit web application that ingests multimodal data (text documents + audio files), indexes them into a vector store (ChromaDB), and delivers retrieval-augmented responses via large language models (LLMs). Responses are further converted into speech output (gTTS) for an interactive user experience.

 Features
•	Text ingestion: Upload PDFs or TXT files → automatic extraction, chunking & embedding with sentence-transformers.
•	Audio ingestion: Upload audio (wav/mp3/m4a/flac) → transcribed with Whisper (local or Hugging Face API).
•	Semantic search: Vectorized storage & retrieval using ChromaDB.
•	LLM response generation:
•	Hugging Face (e.g., GPT-J)
•	OpenAI GPT-3.5 (optional, if API key provided)
•	Voice output: Text-to-speech synthesis with gTTS, playable directly in Streamlit.

Tech Stack
•	Frontend/UI: Streamlit
•	Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
•	Vector DB: ChromaDB
•	LLM Backends: Hugging Face Inference API (GPT-J) / OpenAI GPT-3.5
•	ASR: Whisper (local or HF API)
•	TTS: gTTS
