# main.py
import json
import re
import time
import fitz  # PyMuPDF
import faiss
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple

# Initialize models and constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"  # Change to 'llama2' if preferred
MIN_CHUNK_LENGTH = 30
JSON_PROMPT = """
You are an insurance policy analyst. Determine coverage based on:
1. User query: {query}
2. Relevant policy clauses: {context}

Output JSON format ONLY:
{{
  "decision": "Approved" or "Rejected",
  "reason": "Brief justification (1 sentence)",
  "matched_text": "Exact policy clause text used for decision",
  "clause_reference": "Page X, Section Y",
  "query": "Repeated user query"
}}

Decision Guidelines:
- Approve only if ALL requirements are explicitly satisfied
- Reject if ANY exclusion applies or information is missing
- Never invent details outside provided context
"""

def extract_text_from_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract text chunks with page numbers from PDF"""
    chunks = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            for block in text.split('\n\n'):
                clean_block = re.sub(r'\s+', ' ', block).strip()
                if len(clean_block) >= MIN_CHUNK_LENGTH:
                    chunks.append({
                        "text": clean_block,
                        "page": page_num + 1
                    })
    return chunks

def create_faiss_index(embeddings):
    """Create FAISS index for embeddings"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def query_llm(query: str, context: List[str]) -> Dict[str, Any]:
    """Query local LLM with formatted prompt"""
    context_str = "\n- ".join([f"Page {c['page']}: {c['text']}" for c in context])
    prompt = JSON_PROMPT.format(query=query, context=context_str)
    
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{
            'role': 'user',
            'content': prompt,
            'options': {'temperature': 0.0}
        }]
    )
    return response['message']['content']

def parse_llm_output(llm_output: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Try to parse LLM JSON output with fallback"""
    try:
        start_index = llm_output.find('{')
        end_index = llm_output.rfind('}') + 1
        json_str = llm_output[start_index:end_index]
        return json.loads(json_str), None
    except (json.JSONDecodeError, ValueError) as e:
        return None, llm_output

# Streamlit UI
st.set_page_config(page_title="Policy Analyzer", layout="wide")
st.title("📄 Insurance Policy Analyzer")

# File upload section
uploaded_file = st.file_uploader("Upload Policy PDF", type=["pdf"], key="pdf_uploader")

# Initialize session state
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# Process PDF when uploaded
if uploaded_file is not None:
    with st.spinner("Parsing policy document..."):
        pdf_bytes = uploaded_file.read()
        st.session_state.pdf_chunks = extract_text_from_pdf(pdf_bytes)
        
        # Generate embeddings
        if st.session_state.embedding_model is None:
            st.session_state.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        texts = [chunk["text"] for chunk in st.session_state.pdf_chunks]
        st.session_state.embeddings = st.session_state.embedding_model.encode(texts, show_progress_bar=True)
        st.session_state.faiss_index = create_faiss_index(st.session_state.embeddings)
    
    st.success(f"✅ Processed {len(st.session_state.pdf_chunks)} policy clauses")

# Query section
user_query = st.text_area("Patient/Case Description:", placeholder="46-year-old male, knee surgery in Pune, 3-month-old policy...")

if st.button("Evaluate Claim", disabled=st.session_state.faiss_index is None):
    if not user_query.strip():
        st.warning("Please enter a case description")
        st.stop()
    
    # Semantic search
    with st.spinner("Finding relevant policy clauses..."):
        query_embedding = st.session_state.embedding_model.encode([user_query])
        _, indices = st.session_state.faiss_index.search(query_embedding, k=3)
        top_context = [st.session_state.pdf_chunks[i] for i in indices[0]]
    
    # Query LLM
    with st.spinner("Analyzing with insurance expert..."):
        start_time = time.time()
        llm_output = query_llm(user_query, top_context)
        processing_time = time.time() - start_time
    
    # Parse and display results
    result, raw_output = parse_llm_output(llm_output)
    
    if result:
        st.subheader("Claim Decision")
        st.json(result)
        
        # Display context
        st.divider()
        st.subheader("Relevant Policy Clauses")
        for i, ctx in enumerate(top_context, 1):
            st.markdown(f"**Clause #{i} (Page {ctx['page']}):**")
            st.caption(ctx['text'])
            st.divider()
        
        st.caption(f"Analysis time: {processing_time:.2f} seconds")
    else:
        st.error("Failed to parse structured response")
        st.subheader("Raw LLM Output")
        st.text(raw_output)

# Usage instructions
with st.expander("Usage Instructions"):
    st.markdown("""
    1. **Upload** an insurance policy PDF
    2. **Describe** the medical case in natural language
    3. Click **Evaluate Claim**
    4. View structured decision with policy references

    Features:
    - Semantic search through policy documents
    - Local LLM processing (Mistral via Ollama)
    - Transparent decision reasoning
    - Policy clause verification
    """)
    st.caption("Note: Requires Ollama with Mistral/LLaMA2 installed locally")

# Required installation instructions
st.sidebar.markdown("### Installation Guide")
st.sidebar.code("pip install streamlit pymupdf sentence-transformers faiss-cpu ollama")
st.sidebar.markdown("1. Install [Ollama](https://ollama.ai/)")
st.sidebar.markdown("2. Download model: `ollama pull mistral`")
st.sidebar.markdown("3. Run app: `streamlit run main.py`")