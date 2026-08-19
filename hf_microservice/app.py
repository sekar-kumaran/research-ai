import os
import sys
from pathlib import Path
import gradio as gr
import spaces
import logging

# Ensure src is in the python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from research_ai.configs.settings import Settings
from research_ai.ml_models.classifier.service import ClassifierService
from research_ai.ml_models.summarizer.service import ScientificSummarizer
from research_ai.retrieval.embeddings.service import EmbeddingService
from research_ai.retrieval.hybrid_search.service import HybridSearchService
from research_ai.retrieval.vector_store.faiss_store import FaissVectorStore
from research_ai.ml_models.methodology_extractor.service import MethodologyExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize Models ---
settings = Settings()

logger.info("Initializing ML models for Hugging Face Microservice...")
embedding_service = EmbeddingService(settings.retrieval.embedding_model_name)
vector_store = FaissVectorStore.from_artifacts(settings.paths.similarity_dir)
retriever = HybridSearchService(embedding_service, vector_store)
classifier = ClassifierService.from_artifacts(settings.paths.classifier_dir)
summarizer = ScientificSummarizer()
methodology = MethodologyExtractor()
logger.info("Models initialized successfully.")

# --- Gradio Endpoints (ZeroGPU compatible) ---

@spaces.GPU(duration=60)
def classify_endpoint(title: str, abstract: str):
    try:
        res = classifier.classify(title, abstract)
        return res
    except Exception as e:
        return {"error": str(e)}

@spaces.GPU(duration=60)
def search_endpoint(query: str, top_k: int):
    try:
        # We must load FAISS store on the main thread or ensure it is loaded.
        if not vector_store.ready:
            vector_store._ensure_loaded()
        res = retriever.search(query, top_k=top_k)
        # Gradio API returns JSON strings if we return dicts, which is fine
        return res
    except Exception as e:
        return {"error": str(e)}

@spaces.GPU(duration=120)
def summarize_endpoint(text: str):
    try:
        res = summarizer.summarize(text)
        return {"summary": res}
    except Exception as e:
        return {"error": str(e)}

def methodology_endpoint(text: str):
    try:
        res = methodology.extract(text)
        return res
    except Exception as e:
        return {"error": str(e)}

# --- Build the Gradio UI / API ---
with gr.Blocks(title="Research AI - ML Microservice") as app:
    gr.Markdown("# Research AI - ML Microservice")
    gr.Markdown("This space serves as the ML backend API (Classification, FAISS Search, Summarization).")
    
    with gr.Tab("Status"):
        gr.Markdown(f"✅ **Classifier Loaded**: {classifier.ready}")
        gr.Markdown(f"✅ **FAISS Index Loaded**: {vector_store.ready}")
        
    with gr.Tab("API Testing"):
        with gr.Accordion("Classify"):
            c_title = gr.Textbox(label="Title")
            c_abstract = gr.Textbox(label="Abstract")
            c_btn = gr.Button("Classify")
            c_out = gr.JSON()
            c_btn.click(classify_endpoint, inputs=[c_title, c_abstract], outputs=c_out, api_name="classify")
            
        with gr.Accordion("Search"):
            s_query = gr.Textbox(label="Query")
            s_topk = gr.Number(value=5, label="Top K", precision=0)
            s_btn = gr.Button("Search")
            s_out = gr.JSON()
            s_btn.click(search_endpoint, inputs=[s_query, s_topk], outputs=s_out, api_name="search")
            
        with gr.Accordion("Summarize"):
            sum_text = gr.Textbox(label="Text to summarize", lines=5)
            sum_btn = gr.Button("Summarize")
            sum_out = gr.JSON()
            sum_btn.click(summarize_endpoint, inputs=[sum_text], outputs=sum_out, api_name="summarize")
            
        with gr.Accordion("Methodology"):
            m_text = gr.Textbox(label="Text")
            m_btn = gr.Button("Extract")
            m_out = gr.JSON()
            m_btn.click(methodology_endpoint, inputs=[m_text], outputs=m_out, api_name="methodology")

if __name__ == "__main__":
    app.launch()
