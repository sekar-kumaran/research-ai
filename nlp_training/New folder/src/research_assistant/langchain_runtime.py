from __future__ import annotations

from functools import lru_cache
import os

from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


@lru_cache(maxsize=1)
def get_langchain_llm(model_name: str = "google/flan-t5-small"):
    """Return a cached LangChain LLM wrapper for local or cloud backend."""
    backend = os.getenv("LC_LLM_BACKEND", os.getenv("LLM_BACKEND", "local")).strip().lower()

    if backend == "cloud":
        provider = os.getenv("CLOUD_LLM_PROVIDER", "groq").strip().lower()
        if provider == "groq":
            return ChatOpenAI(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                api_key=os.getenv("GROQ_API_KEY", ""),
                base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                temperature=0.1,
            )
        if provider == "openrouter":
            return ChatOpenAI(
                model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                temperature=0.1,
            )
        if provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except Exception as exc:
                raise ValueError(
                    "Google provider for LangChain requires 'langchain-google-genai'. "
                    "Install it with: pip install langchain-google-genai"
                ) from exc

            return ChatGoogleGenerativeAI(
                model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
                google_api_key=os.getenv("GOOGLE_API_KEY", ""),
                temperature=0.1,
            )
        raise ValueError("Unsupported CLOUD_LLM_PROVIDER for LangChain runtime.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    gen_pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=220,
        do_sample=False,
        truncation=True,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)
