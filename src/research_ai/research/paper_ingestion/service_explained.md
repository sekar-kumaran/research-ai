# service.py Explained

Generated educational companion for `src/research_ai/research/paper_ingestion/service.py`. This file is intentionally detailed so a developer can understand the code, architecture role, production tradeoffs, and ML/backend concepts behind the implementation.

## File Overview

`src/research_ai/research/paper_ingestion/service.py` is a Python module in the Research intelligence layer: paper ingestion, metadata, citations, and trends. It defines PaperChatService and no top-level functions.

## Why This File Exists

This file isolates one responsibility in the codebase: Research intelligence layer: paper ingestion, metadata, citations, and trends. Separation matters because AI systems are easier to test, scale, debug, and explain when retrieval, orchestration, ML services, memory, UI, and deployment scripts have clear boundaries.

## Workflow Position

**Layer:** Research intelligence layer: paper ingestion, metadata, citations, and trends.

**Previous step:** caller code, an API request, a browser event, a test fixture, an import, or a startup script prepares inputs.

**Current step:** `src/research_ai/research/paper_ingestion/service.py` performs its local responsibility.

**Next step:** downstream services, API responses, rendered UI, tests, or process execution consume the result.

```mermaid
flowchart LR
  User[User or Test] --> API[API or Caller]
  API --> ThisFile[src/research_ai/research/paper_ingestion/service.py]
  ThisFile --> Downstream[Downstream Service/UI/Result]
```

## Inputs and Outputs

- **Inputs:** function arguments, class constructor dependencies, HTTP payloads, environment variables, filesystem artifacts, DOM events, or test fixtures.
- **Outputs:** return values, dictionaries, Pydantic models, rendered DOM state, API responses, logs, process startup, assertions, or side effects.
- **Serialization:** this project uses JSON for APIs/LLM planning, parquet/joblib/faiss for ML artifacts, and HTML/CSS/JS for the browser surface.

## Imports Explained

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `io` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `logging` | logging provides structured operational visibility without using print statements. |
| `numpy` | NumPy provides dense numerical arrays used for vector math, similarity computation, normalization, and float32 memory layouts. |
| `os` | os reads environment variables and process/runtime configuration. |
| `pypdf` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `re` | re implements regular expressions for text extraction, validation, and secret redaction. |
| `requests` | Requests is the synchronous HTTP client used for outbound LLM, Ollama, arXiv, or provider calls with explicit timeouts. |
| `research_ai` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `uuid` | uuid creates unique IDs for sessions, conversations, and uploaded-document references. |

## Global Variables and Config

| Name | Line | Why it matters |
|---|---:|---|
| `logger` | 58 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |
| `_ARXIV_VERSION_RE` | 62 | Module-level value, constant, prompt, cache, registry, or configuration point. Check mutability and startup cost. |

## Step-by-Step Workflow

1. Load dependencies and runtime constants.
2. Accept input from the previous layer.
3. Validate, transform, route, score, render, or execute according to this file's role.
4. Return a structured output or perform a controlled side effect.
5. Let caller layers handle presentation, persistence, retries, or fallback.

## Function-by-Function Breakdown

No top-level functions are defined. Behavior is class-based, declarative, or provided through package exports.

## Class-by-Class Breakdown

### `PaperChatService`

- **Line:** 65
- **Base classes:** `object`
- **Docstring:** Full-paper ingestion, chunk retrieval, and grounded paper chat.

**Methods:**
- `__init__` at line 74: method behavior is described by its body and name
- `sessions` at line 93: method behavior is described by its body and name
- `_ensure_generator` at line 96: Lazily initialize the answer-generation backend.

Cloud path: call the injected factory to get the shared singleton client.
  WHY factory not direct import: keeps the singleton pattern consistent
  with the rest of the platform and allows test injection.

Local path: load Flan-T5-small from HuggingFace Hub (seq2seq model).
  WHY Flan-T5-small: small enough to run on CPU, instruction-tuned,
  and produces coherent short answers from context prompts.
- `_build_index` at line 124: method behavior is described by its body and name
- `create_session_from_text` at line 130: method behavior is described by its body and name
- `_extract_pdf_text` at line 157: method behavior is described by its body and name
- `create_session_from_pdf_bytes` at line 161: method behavior is described by its body and name
- `normalize_arxiv_id` at line 168: Normalize any arXiv identifier form to a bare numeric ID.

Handles all common input formats:
  "arxiv:2301.04567"    → "2301.04567"
  "2301.04567v2"        → "2301.04567"   ← BUG FIX: version suffix stripped
  "https://arxiv.org/abs/2301.04567v2" → "2301.04567"
  "https://arxiv.org/pdf/2301.04567.pdf" → "2301.04567"
  "2301.04567"          → "2301.04567"   (pass-through)

WHY strip version suffixes (BUG FIX v3.1.1):
  Without this fix, loading "2301.04567" and "2301.04567v2" would create
  two separate chat sessions for the same paper.  This wastes memory and
  confuses the session cache (find_source finds "arxiv:2301.04567v2" but
  not "arxiv:2301.04567").  By normalizing to the base ID, both resolve
  to the same session.

  The arXiv PDF API serves the latest version when no version is specified,
  so stripping "v2" and fetching the bare ID is safe and gives the same PDF.
- `create_session_from_arxiv_id` at line 213: method behavior is described by its body and name
- `create_or_get_session_from_arxiv_id` at line 222: method behavior is described by its body and name
- `_generate_answer` at line 243: method behavior is described by its body and name
- `ask` at line 262: method behavior is described by its body and name
- `ask_multi` at line 285: method behavior is described by its body and name
- `session_info` at line 328: method behavior is described by its body and name

```python
class PaperChatService:
    """Full-paper ingestion, chunk retrieval, and grounded paper chat."""

    CHAT_SYSTEM = (
        "You are an expert research assistant answering questions about academic papers. "
        "Use only the provided paper context. If evidence is missing, say so. "
        "Do NOT add information from outside the provided context."
    )

    def __init__(
        self,
        embedding_service,
        memory: SessionMemory | None = None,
        cloud_factory=None,
    ) -> None:
        self.embedding_service = embedding_service
        self.memory = memory or SessionMemory()
        self.generator_model_name = "google/flan-t5-small"
        self.backend = os.getenv("LLM_BACKEND", "cloud").strip().lower()
        # cloud_factory: zero-arg callable that returns the shared CloudLLMClient
        # singleton.  None means local-only mode → use Flan-T5 generator.
        self._cloud_factory = cloud_factory
        self._tokenizer = None
        self._generator = None
        # Resolved cloud client (cached after first successful factory call)
        self._cloud = None

    @property
    def sessions(self) -> dict[str, ChatSession]:
        return self.memory.sessions

    def _ensure_generator(self) -> None:
        """Lazily initialize the answer-generation backend.

        Cloud path: call the injected factory to get the shared singleton client.
          WHY factory not direct import: keeps the singleton pattern consistent
          with the rest of the platform and allows test injection.

        Local path: load Flan-T5-small from HuggingFace Hub (seq2seq model).
          WHY Flan-T5-small: small enough to run on CPU, instruction-tuned,
          and produces coherent short answers from context prompts.
        """
        if self.backend == "cloud":
            if self._cloud is None:
                if self._cloud_factory is not None:
                    # Use the shared singleton from the composition root
                    self._cloud = self._cloud_factory()
                else:
                    # Fallback: direct import (only if factory wasn't injected)
                    from research_ai.llm import CloudLLMClient
                    self._cloud = CloudLLMClient()
            return
        # Local Flan-T5 path
        if self._tokenizer is None or self._generator is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
            self._generator = AutoModelForSeq2SeqLM.from_pretrained(self.generator_model_name)

    def _build_index(self, chunks: list[str]):
        vectors = self.embedding_service.encode(chunks).astype(np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def create_session_from_text(
        self,
        text: str,
        source: str,
        title: str = "",
        metadata: dict | None = None,
    ) -> dict:
        chunks = contextual_chunks(text)
        if not chunks:
            raise ValueError("No text content found after chunking.")
        session = ChatSession(
            session_id=str(uuid4()),
            source=source,
            chunks=chunks,
            index=self._build_index(chunks),
            title=title,
            metadata=metadata or {},
        )
        self.memory.put(session)
        return {
            "session_id": session.session_id,
            "source": source,
            "chunk_count": len(chunks),
            "title": title,
        }

    @staticmethod
    def _extract_pdf_text(data: bytes) -> str:
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

    def create_session_from_pdf_bytes(self, data: bytes, source: str) -> dict:
        text = self._extract_pdf_text(data)
        if not text.strip():
            raise ValueError("Could not extract text from PDF. It may be scanned/image-only.")
        return self.create_session_from_text(text=text, source=source)
```

This class packages state and behavior behind a service-style interface. Constructor dependencies show what the class needs; public methods show what the rest of the system can ask it to do. In production, this boundary is where caching, model loading, artifact access, and error handling must be controlled.


## Method-by-Method Deep Dive

### Class `PaperChatService` Methods

#### `PaperChatService.__init__`

- **Line:** 74
- **Kind:** synchronous method
- **Arguments:** self, embedding_service, memory, cloud_factory
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def __init__(
        self,
        embedding_service,
        memory: SessionMemory | None = None,
        cloud_factory=None,
    ) -> None:
        self.embedding_service = embedding_service
        self.memory = memory or SessionMemory()
        self.generator_model_name = "google/flan-t5-small"
        self.backend = os.getenv("LLM_BACKEND", "cloud").strip().lower()
        # cloud_factory: zero-arg callable that returns the shared CloudLLMClient
        # singleton.  None means local-only mode → use Flan-T5 generator.
        self._cloud_factory = cloud_factory
        self._tokenizer = None
        self._generator = None
        # Resolved cloud client (cached after first successful factory call)
        self._cloud = None
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PaperChatService.sessions`

- **Line:** 93
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def sessions(self) -> dict[str, ChatSession]:
        return self.memory.sessions
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PaperChatService._ensure_generator`

- **Line:** 96
- **Kind:** synchronous method
- **Arguments:** self
- **Docstring:** Lazily initialize the answer-generation backend.

Cloud path: call the injected factory to get the shared singleton client.
  WHY factory not direct import: keeps the singleton pattern consistent
  with the rest of the platform and allows test injection.

Local path: load Flan-T5-small from HuggingFace Hub (seq2seq model).
  WHY Flan-T5-small: small enough to run on CPU, instruction-tuned,
  and produces coherent short answers from context prompts.

```python
    def _ensure_generator(self) -> None:
        """Lazily initialize the answer-generation backend.

        Cloud path: call the injected factory to get the shared singleton client.
          WHY factory not direct import: keeps the singleton pattern consistent
          with the rest of the platform and allows test injection.

        Local path: load Flan-T5-small from HuggingFace Hub (seq2seq model).
          WHY Flan-T5-small: small enough to run on CPU, instruction-tuned,
          and produces coherent short answers from context prompts.
        """
        if self.backend == "cloud":
            if self._cloud is None:
                if self._cloud_factory is not None:
                    # Use the shared singleton from the composition root
                    self._cloud = self._cloud_factory()
                else:
                    # Fallback: direct import (only if factory wasn't injected)
                    from research_ai.llm import CloudLLMClient
                    self._cloud = CloudLLMClient()
            return
        # Local Flan-T5 path
        if self._tokenizer is None or self._generator is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
            self._generator = AutoModelForSeq2SeqLM.from_pretrained(self.generator_model_name)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PaperChatService._build_index`

- **Line:** 124
- **Kind:** synchronous method
- **Arguments:** self, chunks
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _build_index(self, chunks: list[str]):
        vectors = self.embedding_service.encode(chunks).astype(np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PaperChatService.create_session_from_text`

- **Line:** 130
- **Kind:** synchronous method
- **Arguments:** self, text, source, title, metadata
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def create_session_from_text(
        self,
        text: str,
        source: str,
        title: str = "",
        metadata: dict | None = None,
    ) -> dict:
        chunks = contextual_chunks(text)
        if not chunks:
            raise ValueError("No text content found after chunking.")
        session = ChatSession(
            session_id=str(uuid4()),
            source=source,
            chunks=chunks,
            index=self._build_index(chunks),
            title=title,
            metadata=metadata or {},
        )
        self.memory.put(session)
        return {
            "session_id": session.session_id,
            "source": source,
            "chunk_count": len(chunks),
            "title": title,
        }
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PaperChatService._extract_pdf_text`

- **Line:** 157
- **Kind:** synchronous method
- **Arguments:** data
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _extract_pdf_text(data: bytes) -> str:
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PaperChatService.create_session_from_pdf_bytes`

- **Line:** 161
- **Kind:** synchronous method
- **Arguments:** self, data, source
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def create_session_from_pdf_bytes(self, data: bytes, source: str) -> dict:
        text = self._extract_pdf_text(data)
        if not text.strip():
            raise ValueError("Could not extract text from PDF. It may be scanned/image-only.")
        return self.create_session_from_text(text=text, source=source)
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PaperChatService.normalize_arxiv_id`

- **Line:** 168
- **Kind:** synchronous method
- **Arguments:** raw_id
- **Docstring:** Normalize any arXiv identifier form to a bare numeric ID.

Handles all common input formats:
  "arxiv:2301.04567"    → "2301.04567"
  "2301.04567v2"        → "2301.04567"   ← BUG FIX: version suffix stripped
  "https://arxiv.org/abs/2301.04567v2" → "2301.04567"
  "https://arxiv.org/pdf/2301.04567.pdf" → "2301.04567"
  "2301.04567"          → "2301.04567"   (pass-through)

WHY strip version suffixes (BUG FIX v3.1.1):
  Without this fix, loading "2301.04567" and "2301.04567v2" would create
  two separate chat sessions for the same paper.  This wastes memory and
  confuses the session cache (find_source finds "arxiv:2301.04567v2" but
  not "arxiv:2301.04567").  By normalizing to the base ID, both resolve
  to the same session.

  The arXiv PDF API serves the latest version when no version is specified,
  so stripping "v2" and fetching the bare ID is safe and gives the same PDF.

```python
    def normalize_arxiv_id(raw_id: str) -> str:
        """Normalize any arXiv identifier form to a bare numeric ID.

        Handles all common input formats:
          "arxiv:2301.04567"    → "2301.04567"
          "2301.04567v2"        → "2301.04567"   ← BUG FIX: version suffix stripped
          "https://arxiv.org/abs/2301.04567v2" → "2301.04567"
          "https://arxiv.org/pdf/2301.04567.pdf" → "2301.04567"
          "2301.04567"          → "2301.04567"   (pass-through)

        WHY strip version suffixes (BUG FIX v3.1.1):
          Without this fix, loading "2301.04567" and "2301.04567v2" would create
          two separate chat sessions for the same paper.  This wastes memory and
          confuses the session cache (find_source finds "arxiv:2301.04567v2" but
          not "arxiv:2301.04567").  By normalizing to the base ID, both resolve
          to the same session.

          The arXiv PDF API serves the latest version when no version is specified,
          so stripping "v2" and fetching the bare ID is safe and gives the same PDF.
        """
        token = (raw_id or "").strip()
        if not token:
            return ""

        # Strip "arxiv:" prefix (case-insensitive)
        if token.lower().startswith("arxiv:"):
            token = token.split(":", 1)[1]

        # Strip URL prefixes — handles abs/, pdf/ paths
        if token.startswith(("http://", "https://")):
            token = token.rstrip("/")
            for marker in ("/abs/", "/pdf/"):
                if marker in token:
                    token = token.split(marker)[-1]
                    break

        # Strip ".pdf" extension
        token = token.replace(".pdf", "").strip()

        # Strip version suffix: "2301.04567v2" → "2301.04567"
        # Applied AFTER all other normalization steps.
        token = _ARXIV_VERSION_RE.sub("", token).strip()

        return token
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PaperChatService.create_session_from_arxiv_id`

- **Line:** 213
- **Kind:** synchronous method
- **Arguments:** self, arxiv_id
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def create_session_from_arxiv_id(self, arxiv_id: str) -> dict:
        normalized = self.normalize_arxiv_id(arxiv_id)
        pdf_url = f"https://arxiv.org/pdf/{normalized}.pdf"
        resp = requests.get(pdf_url, timeout=60, headers={"User-Agent": "ResearchAI/3.0"})
        resp.raise_for_status()
        meta = self.create_session_from_pdf_bytes(resp.content, source=f"arxiv:{normalized}")
        meta["links"] = {"abs": f"https://arxiv.org/abs/{normalized}", "pdf": pdf_url}
        return meta
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PaperChatService.create_or_get_session_from_arxiv_id`

- **Line:** 222
- **Kind:** synchronous method
- **Arguments:** self, arxiv_id
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def create_or_get_session_from_arxiv_id(self, arxiv_id: str) -> dict:
        normalized = self.normalize_arxiv_id(arxiv_id)
        if not normalized:
            raise ValueError(f"Invalid arXiv identifier: '{arxiv_id}'")
        source = f"arxiv:{normalized}"
        existing = self.memory.find_source(source)
        if existing:
            return {
                "session_id": existing.session_id,
                "source": existing.source,
                "chunk_count": len(existing.chunks),
                "cached": True,
                "links": {
                    "abs": f"https://arxiv.org/abs/{normalized}",
                    "pdf": f"https://arxiv.org/pdf/{normalized}.pdf",
                },
            }
        meta = self.create_session_from_arxiv_id(normalized)
        meta["cached"] = False
        return meta
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PaperChatService._generate_answer`

- **Line:** 243
- **Kind:** synchronous method
- **Arguments:** self, question, context, history, max_tokens
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def _generate_answer(self, question: str, context: str, history: list[dict], max_tokens: int = 400) -> str:
        self._ensure_generator()
        if self.backend == "cloud":
            messages: list[dict] = [{"role": "system", "content": self.CHAT_SYSTEM}]
            for turn in history[-3:]:
                messages.append({"role": "user", "content": turn["question"]})
                messages.append({"role": "assistant", "content": turn["answer"]})
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
            return self._cloud.chat(messages, max_tokens=max_tokens)  # type: ignore[union-attr]

        history_text = "\n".join(f"User: {h['question']}\nAssistant: {h['answer']}" for h in history[-3:])
        prompt = (
            "Answer only from the provided paper context.\n\n"
            f"Previous Chat:\n{history_text}\n\nQuestion:\n{question}\n\nPaper Context:\n{context}\n\nAnswer:"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)  # type: ignore[call-arg]
        output_ids = self._generator.generate(**inputs, max_new_tokens=220, do_sample=False)  # type: ignore[union-attr]
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)  # type: ignore[union-attr]
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PaperChatService.ask`

- **Line:** 262
- **Kind:** synchronous method
- **Arguments:** self, session_id, question, top_k
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def ask(self, session_id: str, question: str, top_k: int = 5) -> dict:
        session = self.memory.get(session_id)
        query_vec = self.embedding_service.encode([question]).astype(np.float32)
        scores, ids = session.index.search(query_vec, top_k)
        selected: list[dict] = []
        context_parts: list[str] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk_text = session.chunks[int(idx)]
            selected.append({"chunk_id": int(idx), "score": round(float(score), 4), "text": chunk_text[:800]})
            context_parts.append(chunk_text)
        answer = self._generate_answer(question, "\n\n".join(context_parts), session.history)
        session.history.append({"question": question, "answer": answer})
        return {
            "session_id": session_id,
            "source": session.source,
            "question": question,
            "answer": answer,
            "citations": selected,
            "turns": len(session.history),
        }
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PaperChatService.ask_multi`

- **Line:** 285
- **Kind:** synchronous method
- **Arguments:** self, session_ids, question, top_k_per_session
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def ask_multi(self, session_ids: list[str], question: str, top_k_per_session: int = 3) -> dict:
        if not session_ids:
            raise ValueError("No session IDs provided.")
        collected: list[dict] = []
        for session_id in session_ids:
            try:
                session = self.memory.get(session_id)
            except KeyError:
                logger.warning("Session %s not found, skipping.", session_id)
                continue
            query_vec = self.embedding_service.encode([question]).astype(np.float32)
            scores, ids = session.index.search(query_vec, top_k_per_session)
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0:
                    continue
                collected.append({
                    "session_id": session_id,
                    "source": session.source,
                    "chunk_id": int(idx),
                    "score": float(score),
                    "text": session.chunks[int(idx)],
                })
        if not collected:
            raise ValueError("No retrievable content found across selected sessions.")
        collected.sort(key=lambda item: item["score"], reverse=True)
        top = collected[:10]
        context = "\n\n".join(f"Source: {item['source']}\n{item['text']}" for item in top)
        answer = self._generate_answer(question, context, [])
        return {
            "question": question,
            "answer": answer,
            "citations": [
                {
                    "session_id": item["session_id"],
                    "source": item["source"],
                    "score": round(item["score"], 4),
                    "text": item["text"][:600],
                }
                for item in top
            ],
            "paper_count": len({item["source"] for item in top}),
        }
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

#### `PaperChatService.session_info`

- **Line:** 328
- **Kind:** synchronous method
- **Arguments:** self, session_id
- **Docstring:** No explicit docstring; behavior is inferred from the implementation and call sites.

```python
    def session_info(self, session_id: str) -> dict:
        session = self.memory.get(session_id)
        return {
            "session_id": session.session_id,
            "source": session.source,
            "chunk_count": len(session.chunks),
            "turns": len(session.history),
            "title": session.title,
        }
```

This method is part of the class contract. Its inputs show what state or caller data it consumes; its body shows whether it performs pure transformation, model/artifact access, retrieval, I/O, caching, validation, or orchestration. In production review, pay attention to error handling, mutation of `self`, repeated expensive work, and whether return shapes stay stable for downstream callers.

## Important Algorithms Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Vector Normalization**: Unit-normalized vectors let inner product approximate cosine similarity, a common FAISS retrieval design.
- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Caching**: Caching avoids repeating expensive work such as model loading, embedding generation, or client initialization.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.

## Libraries Used

| Import | Explanation |
|---|---|
| `__future__` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `io` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `logging` | logging provides structured operational visibility without using print statements. |
| `numpy` | NumPy provides dense numerical arrays used for vector math, similarity computation, normalization, and float32 memory layouts. |
| `os` | os reads environment variables and process/runtime configuration. |
| `pypdf` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `re` | re implements regular expressions for text extraction, validation, and secret redaction. |
| `requests` | Requests is the synchronous HTTP client used for outbound LLM, Ollama, arXiv, or provider calls with explicit timeouts. |
| `research_ai` | Project or standard-library dependency used to import a service, type, helper, or runtime capability. Imports affect startup time, packaging, and test isolation. |
| `uuid` | uuid creates unique IDs for sessions, conversations, and uploaded-document references. |

## ML Concepts Used

- **Embeddings**: Embeddings map text into dense semantic vectors so conceptual similarity becomes geometric similarity.
- **Vector Normalization**: Unit-normalized vectors let inner product approximate cosine similarity, a common FAISS retrieval design.
- **FAISS Indexing**: FAISS indexes dense vectors for nearest-neighbor search. Exact flat indexes trade speed at huge scale for simplicity and correctness.
- **RAG**: Retrieval-Augmented Generation retrieves evidence first and asks an LLM to answer from that evidence, reducing hallucination.
- **LLM Inference**: LLM inference sends prompts or chat messages to a model provider and receives generated text under token, latency, and cost constraints.
- **Transformers**: Transformers use tokenization and attention layers for language understanding/generation. They are powerful but memory and latency sensitive.
- **Caching**: Caching avoids repeating expensive work such as model loading, embedding generation, or client initialization.
- **Sandboxing**: Sandboxing validates and constrains user code before execution, reducing security and stability risk.

## Performance and Memory Notes

- Avoid eager loading of heavy ML models unless startup latency is acceptable.
- Cache expensive clients, tokenizers, vector stores, and embeddings carefully.
- Use float32 for embedding vectors because it halves memory compared with float64 and matches FAISS/neural inference expectations.
- Bound prompt length, uploaded content, result counts, and token budgets to control latency and memory.
- Watch copies of large metadata frames, embedding matrices, and file buffers.

## Scalability Notes

- In-memory state works for local demos but needs Redis/database/object storage for multi-worker cloud deployment.
- CPU/GPU inference should often be separated from the web API when traffic grows.
- Retrieval can start exact and move to approximate indexes as corpus size grows.
- Batch operations and cache repeated work to improve throughput.
- Add metrics for latency, errors, fallback frequency, retrieval hit rate, and token usage.

## Production Engineering Notes

- Keep interfaces stable because other files may import this module or depend on its response shape.
- Prefer typed/structured data over free-form strings at service boundaries.
- Log operational context without secrets or huge payloads.
- Make fallback behavior explicit so users get useful output even when LLMs or artifacts fail.
- Keep provider-specific logic behind adapters so Groq/OpenRouter/Google/Ollama can be swapped.

## Common Bugs and Failure Cases

- Missing `.env` values, model artifacts, or Ollama models can trigger degraded behavior.
- Type mismatches occur when LLM-generated tool arguments cross into strict Python code.
- Empty retrieval results must not become hallucinated answers.
- Network calls need timeouts and careful retry behavior.
- Frontend IDs/classes and API schemas are contracts; changing one side without the other breaks workflows.

## Security Considerations

- Handles credentials or environment configuration. Keep secrets in environment variables and redact them from logs.
- Deals with execution or subprocesses. Maintain AST validation, isolated mode, timeouts, and least privilege.
- Performs network I/O. Use timeouts, validate responses, and keep private services such as Ollama off the public internet.

## Real Industry Usage

- This pattern appears in enterprise RAG assistants, scientific search tools, internal research copilots, and ML platform demos.
- The layered design mirrors production systems: API facade, orchestration, retrieval, evaluation, synthesis, UI, and deployment.
- Clear separation lets teams replace model providers, improve retrieval, harden security, or redesign UI independently.

## Optimization Opportunities

- Add tracing around each workflow step.
- Strengthen schema validation at boundaries.
- Persist conversation/session state outside process memory.
- Add load tests and adversarial tests for prompt injection, empty evidence, and large uploads.
- Consider approximate vector indexes, reranker models, or batching when corpus/traffic grows.

## How This Connects To Other Files

- `src/research_ai/research/paper_ingestion/service.py` is connected through imports, startup scripts, API routes, frontend selectors, tests, or artifact paths.
- `src/research_ai/platform.py` is the backend composition root.
- `src/research_ai/api/main.py` exposes backend behavior over HTTP.
- Retrieval modules depend on artifacts under `artifacts/`.
- Frontend files depend on stable endpoint and DOM contracts.

## End-to-End Flow Summary

- A user/browser/test/startup event enters the system.
- The relevant layer validates or normalizes input.
- Retrieval, ML, orchestration, execution, or UI rendering happens.
- A structured result, visual state, or process side effect is produced.
- Fallbacks and tests keep behavior understandable when dependencies are unavailable.

## Interview Questions

1. What responsibility does this file own?
2. What inputs and outputs define its contract?
3. Which dependencies are expensive or operationally risky?
4. What breaks if this file changes shape?
5. How would you scale or test this behavior in production?

## Key Takeaways

- `src/research_ai/research/paper_ingestion/service.py` should be understood as part of a layered AI research platform.
- Trace data flow from inputs to transformations to outputs.
- Production readiness comes from explicit contracts, bounded resources, observability, secure defaults, and graceful fallback.

## Fully Commented Source

This section repeats the original source with an explanatory comment before every line. The comments are educational only; they are not inserted into the production source file.

```python
# L0001: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""Full-paper ingestion, per-session FAISS chunk retrieval, and grounded paper chat.
# L0002: Blank line that visually separates logical sections and improves readability.

# L0003: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
ARCHITECTURE
# L0004: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
------------
# L0005: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Each paper loaded into the system (via PDF, text, or arXiv ID) gets its own
# L0006: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
mini-FAISS index (IndexFlatIP) built from its chunk embeddings.  This index is
# L0007: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
stored in a ChatSession alongside the chunk text and conversation history.
# L0008: Blank line that visually separates logical sections and improves readability.

# L0009: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
WHY a per-paper FAISS index?
# L0010: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  - The global FAISS index only has title+abstract embeddings (one vector/paper).
# L0011: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  - Full-paper chat needs section-level retrieval: finding which passage of a
# L0012: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    50-page paper answers a specific question.
# L0013: Assigns or updates a value used later in the workflow; check mutability and data shape.
  - Per-session indices are tiny (10–100 chunks × 384 dims = <1MB each) and
# L0014: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    allow session isolation — querying one paper never pollutes another.
# L0015: Blank line that visually separates logical sections and improves readability.

# L0016: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
CHUNKING STRATEGY
# L0017: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
-----------------
# L0018: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
contextual_chunks() (see retrieval/chunking.py) splits text into 900-word
# L0019: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
sentence-boundary-respecting chunks with 150-word overlap.  The overlap means
# L0020: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
that a sentence at the boundary of a chunk appears in both adjacent chunks,
# L0021: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
so context is never lost at retrieval time.
# L0022: Blank line that visually separates logical sections and improves readability.

# L0023: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
RETRIEVAL IN ask()
# L0024: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
------------------
# L0025: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  1. Embed the user question with EmbeddingService (L2-normalised).
# L0026: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  2. IndexFlatIP.search() → top_k chunks by cosine similarity.
# L0027: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  3. Concatenate chunk texts as context for the LLM.
# L0028: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
  4. LLM generates a grounded answer citing only the provided context.
# L0029: Blank line that visually separates logical sections and improves readability.

# L0030: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
CLOUD FACTORY INJECTION (BUG FIX v3.1.1)
# L0031: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
-----------------------------------------
# L0032: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Previously this class called CloudLLMClient() directly, creating a second
# L0033: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
client instance and breaking the singleton pattern established in platform.py.
# L0034: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
Fix: accept a cloud_factory callable from the composition root.  The factory
# L0035: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
returns the shared singleton.  If cloud_factory is None (local-only mode), the
# L0036: Defines a class that groups related state and behavior behind a reusable interface.
class falls back to the local Flan-T5 generator.
# L0037: Starts, ends, or continues a docstring that documents module, class, or function intent.
"""
# L0038: Enables future Python behavior so annotations/import semantics stay modern and predictable.
from __future__ import annotations
# L0039: Blank line that visually separates logical sections and improves readability.

# L0040: Imports a dependency, type, or project module needed by later code in this file.
import logging
# L0041: Imports a dependency, type, or project module needed by later code in this file.
import os
# L0042: Imports a dependency, type, or project module needed by later code in this file.
import re
# L0043: Imports a dependency, type, or project module needed by later code in this file.
from io import BytesIO
# L0044: Imports a dependency, type, or project module needed by later code in this file.
from uuid import uuid4
# L0045: Blank line that visually separates logical sections and improves readability.

# L0046: Imports a dependency, type, or project module needed by later code in this file.
import numpy as np
# L0047: Imports a dependency, type, or project module needed by later code in this file.
import requests
# L0048: Imports a dependency, type, or project module needed by later code in this file.
from pypdf import PdfReader
# L0049: Blank line that visually separates logical sections and improves readability.

# L0050: Begins protected execution so failures can be handled without crashing the whole request path.
try:
# L0051: Imports a dependency, type, or project module needed by later code in this file.
    import faiss
# L0052: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
except ImportError as exc:  # pragma: no cover
# L0053: Raises an explicit error when the function cannot safely continue.
    raise RuntimeError("faiss-cpu is required. Install with: pip install faiss-cpu") from exc
# L0054: Blank line that visually separates logical sections and improves readability.

# L0055: Imports a dependency, type, or project module needed by later code in this file.
from research_ai.memory.session_memory import ChatSession, SessionMemory
# L0056: Imports a dependency, type, or project module needed by later code in this file.
from research_ai.retrieval.chunking import contextual_chunks
# L0057: Blank line that visually separates logical sections and improves readability.

# L0058: Assigns or updates a value used later in the workflow; check mutability and data shape.
logger = logging.getLogger(__name__)
# L0059: Blank line that visually separates logical sections and improves readability.

# L0060: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# ArXiv version suffix pattern: matches "v1", "v2", "v10", etc. at end of ID.
# L0061: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
# Example: "2301.04567v2" → "2301.04567"
# L0062: Assigns or updates a value used later in the workflow; check mutability and data shape.
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
# L0063: Blank line that visually separates logical sections and improves readability.

# L0064: Blank line that visually separates logical sections and improves readability.

# L0065: Defines a class that groups related state and behavior behind a reusable interface.
class PaperChatService:
# L0066: Starts, ends, or continues a docstring that documents module, class, or function intent.
    """Full-paper ingestion, chunk retrieval, and grounded paper chat."""
# L0067: Blank line that visually separates logical sections and improves readability.

# L0068: Assigns or updates a value used later in the workflow; check mutability and data shape.
    CHAT_SYSTEM = (
# L0069: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "You are an expert research assistant answering questions about academic papers. "
# L0070: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "Use only the provided paper context. If evidence is missing, say so. "
# L0071: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        "Do NOT add information from outside the provided context."
# L0072: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    )
# L0073: Blank line that visually separates logical sections and improves readability.

# L0074: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def __init__(
# L0075: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        self,
# L0076: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        embedding_service,
# L0077: Assigns or updates a value used later in the workflow; check mutability and data shape.
        memory: SessionMemory | None = None,
# L0078: Assigns or updates a value used later in the workflow; check mutability and data shape.
        cloud_factory=None,
# L0079: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ) -> None:
# L0080: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.embedding_service = embedding_service
# L0081: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.memory = memory or SessionMemory()
# L0082: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.generator_model_name = "google/flan-t5-small"
# L0083: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self.backend = os.getenv("LLM_BACKEND", "cloud").strip().lower()
# L0084: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # cloud_factory: zero-arg callable that returns the shared CloudLLMClient
# L0085: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # singleton.  None means local-only mode → use Flan-T5 generator.
# L0086: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self._cloud_factory = cloud_factory
# L0087: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self._tokenizer = None
# L0088: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self._generator = None
# L0089: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Resolved cloud client (cached after first successful factory call)
# L0090: Assigns or updates a value used later in the workflow; check mutability and data shape.
        self._cloud = None
# L0091: Blank line that visually separates logical sections and improves readability.

# L0092: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @property
# L0093: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def sessions(self) -> dict[str, ChatSession]:
# L0094: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return self.memory.sessions
# L0095: Blank line that visually separates logical sections and improves readability.

# L0096: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _ensure_generator(self) -> None:
# L0097: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Lazily initialize the answer-generation backend.
# L0098: Blank line that visually separates logical sections and improves readability.

# L0099: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Cloud path: call the injected factory to get the shared singleton client.
# L0100: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          WHY factory not direct import: keeps the singleton pattern consistent
# L0101: Uses a context manager to guarantee setup/cleanup around files, locks, or managed resources.
          with the rest of the platform and allows test injection.
# L0102: Blank line that visually separates logical sections and improves readability.

# L0103: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Local path: load Flan-T5-small from HuggingFace Hub (seq2seq model).
# L0104: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          WHY Flan-T5-small: small enough to run on CPU, instruction-tuned,
# L0105: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          and produces coherent short answers from context prompts.
# L0106: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0107: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self.backend == "cloud":
# L0108: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if self._cloud is None:
# L0109: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if self._cloud_factory is not None:
# L0110: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
                    # Use the shared singleton from the composition root
# L0111: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    self._cloud = self._cloud_factory()
# L0112: Continues conditional control flow for alternate cases or default fallback behavior.
                else:
# L0113: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
                    # Fallback: direct import (only if factory wasn't injected)
# L0114: Imports a dependency, type, or project module needed by later code in this file.
                    from research_ai.llm import CloudLLMClient
# L0115: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    self._cloud = CloudLLMClient()
# L0116: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            return
# L0117: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Local Flan-T5 path
# L0118: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self._tokenizer is None or self._generator is None:
# L0119: Imports a dependency, type, or project module needed by later code in this file.
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# L0120: Blank line that visually separates logical sections and improves readability.

# L0121: Assigns or updates a value used later in the workflow; check mutability and data shape.
            self._tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
# L0122: Assigns or updates a value used later in the workflow; check mutability and data shape.
            self._generator = AutoModelForSeq2SeqLM.from_pretrained(self.generator_model_name)
# L0123: Blank line that visually separates logical sections and improves readability.

# L0124: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _build_index(self, chunks: list[str]):
# L0125: Assigns or updates a value used later in the workflow; check mutability and data shape.
        vectors = self.embedding_service.encode(chunks).astype(np.float32)
# L0126: Assigns or updates a value used later in the workflow; check mutability and data shape.
        index = faiss.IndexFlatIP(vectors.shape[1])
# L0127: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        index.add(vectors)
# L0128: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return index
# L0129: Blank line that visually separates logical sections and improves readability.

# L0130: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def create_session_from_text(
# L0131: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        self,
# L0132: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        text: str,
# L0133: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        source: str,
# L0134: Assigns or updates a value used later in the workflow; check mutability and data shape.
        title: str = "",
# L0135: Assigns or updates a value used later in the workflow; check mutability and data shape.
        metadata: dict | None = None,
# L0136: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
    ) -> dict:
# L0137: Assigns or updates a value used later in the workflow; check mutability and data shape.
        chunks = contextual_chunks(text)
# L0138: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not chunks:
# L0139: Raises an explicit error when the function cannot safely continue.
            raise ValueError("No text content found after chunking.")
# L0140: Assigns or updates a value used later in the workflow; check mutability and data shape.
        session = ChatSession(
# L0141: Assigns or updates a value used later in the workflow; check mutability and data shape.
            session_id=str(uuid4()),
# L0142: Assigns or updates a value used later in the workflow; check mutability and data shape.
            source=source,
# L0143: Assigns or updates a value used later in the workflow; check mutability and data shape.
            chunks=chunks,
# L0144: Assigns or updates a value used later in the workflow; check mutability and data shape.
            index=self._build_index(chunks),
# L0145: Assigns or updates a value used later in the workflow; check mutability and data shape.
            title=title,
# L0146: Assigns or updates a value used later in the workflow; check mutability and data shape.
            metadata=metadata or {},
# L0147: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0148: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        self.memory.put(session)
# L0149: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return {
# L0150: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "session_id": session.session_id,
# L0151: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "source": source,
# L0152: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "chunk_count": len(chunks),
# L0153: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "title": title,
# L0154: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0155: Blank line that visually separates logical sections and improves readability.

# L0156: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @staticmethod
# L0157: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _extract_pdf_text(data: bytes) -> str:
# L0158: Assigns or updates a value used later in the workflow; check mutability and data shape.
        reader = PdfReader(BytesIO(data))
# L0159: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
# L0160: Blank line that visually separates logical sections and improves readability.

# L0161: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def create_session_from_pdf_bytes(self, data: bytes, source: str) -> dict:
# L0162: Assigns or updates a value used later in the workflow; check mutability and data shape.
        text = self._extract_pdf_text(data)
# L0163: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not text.strip():
# L0164: Raises an explicit error when the function cannot safely continue.
            raise ValueError("Could not extract text from PDF. It may be scanned/image-only.")
# L0165: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return self.create_session_from_text(text=text, source=source)
# L0166: Blank line that visually separates logical sections and improves readability.

# L0167: Decorator that modifies the function/class below, commonly for routing, dataclasses, caching, or properties.
    @staticmethod
# L0168: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def normalize_arxiv_id(raw_id: str) -> str:
# L0169: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """Normalize any arXiv identifier form to a bare numeric ID.
# L0170: Blank line that visually separates logical sections and improves readability.

# L0171: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        Handles all common input formats:
# L0172: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          "arxiv:2301.04567"    → "2301.04567"
# L0173: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          "2301.04567v2"        → "2301.04567"   ← BUG FIX: version suffix stripped
# L0174: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          "https://arxiv.org/abs/2301.04567v2" → "2301.04567"
# L0175: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          "https://arxiv.org/pdf/2301.04567.pdf" → "2301.04567"
# L0176: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          "2301.04567"          → "2301.04567"   (pass-through)
# L0177: Blank line that visually separates logical sections and improves readability.

# L0178: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        WHY strip version suffixes (BUG FIX v3.1.1):
# L0179: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          Without this fix, loading "2301.04567" and "2301.04567v2" would create
# L0180: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          two separate chat sessions for the same paper.  This wastes memory and
# L0181: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          confuses the session cache (find_source finds "arxiv:2301.04567v2" but
# L0182: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          not "arxiv:2301.04567").  By normalizing to the base ID, both resolve
# L0183: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          to the same session.
# L0184: Blank line that visually separates logical sections and improves readability.

# L0185: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          The arXiv PDF API serves the latest version when no version is specified,
# L0186: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
          so stripping "v2" and fetching the bare ID is safe and gives the same PDF.
# L0187: Starts, ends, or continues a docstring that documents module, class, or function intent.
        """
# L0188: Assigns or updates a value used later in the workflow; check mutability and data shape.
        token = (raw_id or "").strip()
# L0189: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not token:
# L0190: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return ""
# L0191: Blank line that visually separates logical sections and improves readability.

# L0192: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Strip "arxiv:" prefix (case-insensitive)
# L0193: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if token.lower().startswith("arxiv:"):
# L0194: Assigns or updates a value used later in the workflow; check mutability and data shape.
            token = token.split(":", 1)[1]
# L0195: Blank line that visually separates logical sections and improves readability.

# L0196: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Strip URL prefixes — handles abs/, pdf/ paths
# L0197: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if token.startswith(("http://", "https://")):
# L0198: Assigns or updates a value used later in the workflow; check mutability and data shape.
            token = token.rstrip("/")
# L0199: Iterates over data, retry attempts, files, results, or workflow steps.
            for marker in ("/abs/", "/pdf/"):
# L0200: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if marker in token:
# L0201: Assigns or updates a value used later in the workflow; check mutability and data shape.
                    token = token.split(marker)[-1]
# L0202: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    break
# L0203: Blank line that visually separates logical sections and improves readability.

# L0204: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Strip ".pdf" extension
# L0205: Assigns or updates a value used later in the workflow; check mutability and data shape.
        token = token.replace(".pdf", "").strip()
# L0206: Blank line that visually separates logical sections and improves readability.

# L0207: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Strip version suffix: "2301.04567v2" → "2301.04567"
# L0208: Developer comment explaining intent, caveat, section boundary, or operational reasoning.
        # Applied AFTER all other normalization steps.
# L0209: Assigns or updates a value used later in the workflow; check mutability and data shape.
        token = _ARXIV_VERSION_RE.sub("", token).strip()
# L0210: Blank line that visually separates logical sections and improves readability.

# L0211: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return token
# L0212: Blank line that visually separates logical sections and improves readability.

# L0213: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def create_session_from_arxiv_id(self, arxiv_id: str) -> dict:
# L0214: Assigns or updates a value used later in the workflow; check mutability and data shape.
        normalized = self.normalize_arxiv_id(arxiv_id)
# L0215: Assigns or updates a value used later in the workflow; check mutability and data shape.
        pdf_url = f"https://arxiv.org/pdf/{normalized}.pdf"
# L0216: Assigns or updates a value used later in the workflow; check mutability and data shape.
        resp = requests.get(pdf_url, timeout=60, headers={"User-Agent": "ResearchAI/3.0"})
# L0217: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        resp.raise_for_status()
# L0218: Assigns or updates a value used later in the workflow; check mutability and data shape.
        meta = self.create_session_from_pdf_bytes(resp.content, source=f"arxiv:{normalized}")
# L0219: Assigns or updates a value used later in the workflow; check mutability and data shape.
        meta["links"] = {"abs": f"https://arxiv.org/abs/{normalized}", "pdf": pdf_url}
# L0220: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return meta
# L0221: Blank line that visually separates logical sections and improves readability.

# L0222: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def create_or_get_session_from_arxiv_id(self, arxiv_id: str) -> dict:
# L0223: Assigns or updates a value used later in the workflow; check mutability and data shape.
        normalized = self.normalize_arxiv_id(arxiv_id)
# L0224: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not normalized:
# L0225: Raises an explicit error when the function cannot safely continue.
            raise ValueError(f"Invalid arXiv identifier: '{arxiv_id}'")
# L0226: Assigns or updates a value used later in the workflow; check mutability and data shape.
        source = f"arxiv:{normalized}"
# L0227: Assigns or updates a value used later in the workflow; check mutability and data shape.
        existing = self.memory.find_source(source)
# L0228: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if existing:
# L0229: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return {
# L0230: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "session_id": existing.session_id,
# L0231: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "source": existing.source,
# L0232: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "chunk_count": len(existing.chunks),
# L0233: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "cached": True,
# L0234: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                "links": {
# L0235: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "abs": f"https://arxiv.org/abs/{normalized}",
# L0236: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "pdf": f"https://arxiv.org/pdf/{normalized}.pdf",
# L0237: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                },
# L0238: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            }
# L0239: Assigns or updates a value used later in the workflow; check mutability and data shape.
        meta = self.create_session_from_arxiv_id(normalized)
# L0240: Assigns or updates a value used later in the workflow; check mutability and data shape.
        meta["cached"] = False
# L0241: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return meta
# L0242: Blank line that visually separates logical sections and improves readability.

# L0243: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def _generate_answer(self, question: str, context: str, history: list[dict], max_tokens: int = 400) -> str:
# L0244: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        self._ensure_generator()
# L0245: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if self.backend == "cloud":
# L0246: Assigns or updates a value used later in the workflow; check mutability and data shape.
            messages: list[dict] = [{"role": "system", "content": self.CHAT_SYSTEM}]
# L0247: Iterates over data, retry attempts, files, results, or workflow steps.
            for turn in history[-3:]:
# L0248: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                messages.append({"role": "user", "content": turn["question"]})
# L0249: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                messages.append({"role": "assistant", "content": turn["answer"]})
# L0250: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
# L0251: Returns the computed result to the caller; this shape becomes part of the downstream contract.
            return self._cloud.chat(messages, max_tokens=max_tokens)  # type: ignore[union-attr]
# L0252: Blank line that visually separates logical sections and improves readability.

# L0253: Assigns or updates a value used later in the workflow; check mutability and data shape.
        history_text = "\n".join(f"User: {h['question']}\nAssistant: {h['answer']}" for h in history[-3:])
# L0254: Assigns or updates a value used later in the workflow; check mutability and data shape.
        prompt = (
# L0255: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "Answer only from the provided paper context.\n\n"
# L0256: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            f"Previous Chat:\n{history_text}\n\nQuestion:\n{question}\n\nPaper Context:\n{context}\n\nAnswer:"
# L0257: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        )
# L0258: Assigns or updates a value used later in the workflow; check mutability and data shape.
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)  # type: ignore[call-arg]
# L0259: Assigns or updates a value used later in the workflow; check mutability and data shape.
        output_ids = self._generator.generate(**inputs, max_new_tokens=220, do_sample=False)  # type: ignore[union-attr]
# L0260: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)  # type: ignore[union-attr]
# L0261: Blank line that visually separates logical sections and improves readability.

# L0262: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def ask(self, session_id: str, question: str, top_k: int = 5) -> dict:
# L0263: Assigns or updates a value used later in the workflow; check mutability and data shape.
        session = self.memory.get(session_id)
# L0264: Assigns or updates a value used later in the workflow; check mutability and data shape.
        query_vec = self.embedding_service.encode([question]).astype(np.float32)
# L0265: Assigns or updates a value used later in the workflow; check mutability and data shape.
        scores, ids = session.index.search(query_vec, top_k)
# L0266: Assigns or updates a value used later in the workflow; check mutability and data shape.
        selected: list[dict] = []
# L0267: Assigns or updates a value used later in the workflow; check mutability and data shape.
        context_parts: list[str] = []
# L0268: Iterates over data, retry attempts, files, results, or workflow steps.
        for score, idx in zip(scores[0], ids[0]):
# L0269: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
            if idx < 0:
# L0270: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                continue
# L0271: Assigns or updates a value used later in the workflow; check mutability and data shape.
            chunk_text = session.chunks[int(idx)]
# L0272: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            selected.append({"chunk_id": int(idx), "score": round(float(score), 4), "text": chunk_text[:800]})
# L0273: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            context_parts.append(chunk_text)
# L0274: Assigns or updates a value used later in the workflow; check mutability and data shape.
        answer = self._generate_answer(question, "\n\n".join(context_parts), session.history)
# L0275: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        session.history.append({"question": question, "answer": answer})
# L0276: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return {
# L0277: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "session_id": session_id,
# L0278: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "source": session.source,
# L0279: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "question": question,
# L0280: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "answer": answer,
# L0281: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "citations": selected,
# L0282: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "turns": len(session.history),
# L0283: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0284: Blank line that visually separates logical sections and improves readability.

# L0285: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def ask_multi(self, session_ids: list[str], question: str, top_k_per_session: int = 3) -> dict:
# L0286: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not session_ids:
# L0287: Raises an explicit error when the function cannot safely continue.
            raise ValueError("No session IDs provided.")
# L0288: Assigns or updates a value used later in the workflow; check mutability and data shape.
        collected: list[dict] = []
# L0289: Iterates over data, retry attempts, files, results, or workflow steps.
        for session_id in session_ids:
# L0290: Begins protected execution so failures can be handled without crashing the whole request path.
            try:
# L0291: Assigns or updates a value used later in the workflow; check mutability and data shape.
                session = self.memory.get(session_id)
# L0292: Handles an expected failure path, often converting exceptions into fallback behavior or API errors.
            except KeyError:
# L0293: Emits structured operational information for debugging, monitoring, or failure diagnosis.
                logger.warning("Session %s not found, skipping.", session_id)
# L0294: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                continue
# L0295: Assigns or updates a value used later in the workflow; check mutability and data shape.
            query_vec = self.embedding_service.encode([question]).astype(np.float32)
# L0296: Assigns or updates a value used later in the workflow; check mutability and data shape.
            scores, ids = session.index.search(query_vec, top_k_per_session)
# L0297: Iterates over data, retry attempts, files, results, or workflow steps.
            for score, idx in zip(scores[0], ids[0]):
# L0298: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
                if idx < 0:
# L0299: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    continue
# L0300: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                collected.append({
# L0301: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "session_id": session_id,
# L0302: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "source": session.source,
# L0303: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "chunk_id": int(idx),
# L0304: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "score": float(score),
# L0305: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "text": session.chunks[int(idx)],
# L0306: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                })
# L0307: Branches execution based on a condition, usually for validation, fallback, or mode-specific behavior.
        if not collected:
# L0308: Raises an explicit error when the function cannot safely continue.
            raise ValueError("No retrievable content found across selected sessions.")
# L0309: Assigns or updates a value used later in the workflow; check mutability and data shape.
        collected.sort(key=lambda item: item["score"], reverse=True)
# L0310: Assigns or updates a value used later in the workflow; check mutability and data shape.
        top = collected[:10]
# L0311: Assigns or updates a value used later in the workflow; check mutability and data shape.
        context = "\n\n".join(f"Source: {item['source']}\n{item['text']}" for item in top)
# L0312: Assigns or updates a value used later in the workflow; check mutability and data shape.
        answer = self._generate_answer(question, context, [])
# L0313: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return {
# L0314: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "question": question,
# L0315: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "answer": answer,
# L0316: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "citations": [
# L0317: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                {
# L0318: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "session_id": item["session_id"],
# L0319: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "source": item["source"],
# L0320: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "score": round(item["score"], 4),
# L0321: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                    "text": item["text"][:600],
# L0322: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
                }
# L0323: Iterates over data, retry attempts, files, results, or workflow steps.
                for item in top
# L0324: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            ],
# L0325: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "paper_count": len({item["source"] for item in top}),
# L0326: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0327: Blank line that visually separates logical sections and improves readability.

# L0328: Defines a function or method; parameters are the input contract and the body implements the workflow.
    def session_info(self, session_id: str) -> dict:
# L0329: Assigns or updates a value used later in the workflow; check mutability and data shape.
        session = self.memory.get(session_id)
# L0330: Returns the computed result to the caller; this shape becomes part of the downstream contract.
        return {
# L0331: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "session_id": session.session_id,
# L0332: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "source": session.source,
# L0333: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "chunk_count": len(session.chunks),
# L0334: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "turns": len(session.history),
# L0335: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
            "title": session.title,
# L0336: Executes Python logic as part of this file's workflow; read surrounding lines for data flow and side effects.
        }
# L0337: Blank line that visually separates logical sections and improves readability.

```

## Source Walkthrough

This file is large, so the opening and closing sections are included here. Use the class/function breakdown above to navigate the middle of the file.

### Opening Section

```python
"""Full-paper ingestion, per-session FAISS chunk retrieval, and grounded paper chat.

ARCHITECTURE
------------
Each paper loaded into the system (via PDF, text, or arXiv ID) gets its own
mini-FAISS index (IndexFlatIP) built from its chunk embeddings.  This index is
stored in a ChatSession alongside the chunk text and conversation history.

WHY a per-paper FAISS index?
  - The global FAISS index only has title+abstract embeddings (one vector/paper).
  - Full-paper chat needs section-level retrieval: finding which passage of a
    50-page paper answers a specific question.
  - Per-session indices are tiny (10–100 chunks × 384 dims = <1MB each) and
    allow session isolation — querying one paper never pollutes another.

CHUNKING STRATEGY
-----------------
contextual_chunks() (see retrieval/chunking.py) splits text into 900-word
sentence-boundary-respecting chunks with 150-word overlap.  The overlap means
that a sentence at the boundary of a chunk appears in both adjacent chunks,
so context is never lost at retrieval time.

RETRIEVAL IN ask()
------------------
  1. Embed the user question with EmbeddingService (L2-normalised).
  2. IndexFlatIP.search() → top_k chunks by cosine similarity.
  3. Concatenate chunk texts as context for the LLM.
  4. LLM generates a grounded answer citing only the provided context.

CLOUD FACTORY INJECTION (BUG FIX v3.1.1)
-----------------------------------------
Previously this class called CloudLLMClient() directly, creating a second
client instance and breaking the singleton pattern established in platform.py.
Fix: accept a cloud_factory callable from the composition root.  The factory
returns the shared singleton.  If cloud_factory is None (local-only mode), the
class falls back to the local Flan-T5 generator.
"""
from __future__ import annotations

import logging
import os
import re
from io import BytesIO
from uuid import uuid4

import numpy as np
import requests
from pypdf import PdfReader

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("faiss-cpu is required. Install with: pip install faiss-cpu") from exc

from research_ai.memory.session_memory import ChatSession, SessionMemory
from research_ai.retrieval.chunking import contextual_chunks

logger = logging.getLogger(__name__)

# ArXiv version suffix pattern: matches "v1", "v2", "v10", etc. at end of ID.
# Example: "2301.04567v2" → "2301.04567"
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


class PaperChatService:
    """Full-paper ingestion, chunk retrieval, and grounded paper chat."""

    CHAT_SYSTEM = (
        "You are an expert research assistant answering questions about academic papers. "
        "Use only the provided paper context. If evidence is missing, say so. "
        "Do NOT add information from outside the provided context."
    )

    def __init__(
        self,
        embedding_service,
        memory: SessionMemory | None = None,
        cloud_factory=None,
    ) -> None:
        self.embedding_service = embedding_service
        self.memory = memory or SessionMemory()
        self.generator_model_name = "google/flan-t5-small"
        self.backend = os.getenv("LLM_BACKEND", "cloud").strip().lower()
        # cloud_factory: zero-arg callable that returns the shared CloudLLMClient
        # singleton.  None means local-only mode → use Flan-T5 generator.
        self._cloud_factory = cloud_factory
        self._tokenizer = None
        self._generator = None
        # Resolved cloud client (cached after first successful factory call)
        self._cloud = None

    @property
    def sessions(self) -> dict[str, ChatSession]:
        return self.memory.sessions

    def _ensure_generator(self) -> None:
        """Lazily initialize the answer-generation backend.

        Cloud path: call the injected factory to get the shared singleton client.
          WHY factory not direct import: keeps the singleton pattern consistent
          with the rest of the platform and allows test injection.

        Local path: load Flan-T5-small from HuggingFace Hub (seq2seq model).
          WHY Flan-T5-small: small enough to run on CPU, instruction-tuned,
          and produces coherent short answers from context prompts.
        """
        if self.backend == "cloud":
            if self._cloud is None:
                if self._cloud_factory is not None:
                    # Use the shared singleton from the composition root
                    self._cloud = self._cloud_factory()
                else:
                    # Fallback: direct import (only if factory wasn't injected)
                    from research_ai.llm import CloudLLMClient
                    self._cloud = CloudLLMClient()
            return
        # Local Flan-T5 path
        if self._tokenizer is None or self._generator is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
```

### Closing Section

```python
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)  # type: ignore[call-arg]
        output_ids = self._generator.generate(**inputs, max_new_tokens=220, do_sample=False)  # type: ignore[union-attr]
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)  # type: ignore[union-attr]

    def ask(self, session_id: str, question: str, top_k: int = 5) -> dict:
        session = self.memory.get(session_id)
        query_vec = self.embedding_service.encode([question]).astype(np.float32)
        scores, ids = session.index.search(query_vec, top_k)
        selected: list[dict] = []
        context_parts: list[str] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk_text = session.chunks[int(idx)]
            selected.append({"chunk_id": int(idx), "score": round(float(score), 4), "text": chunk_text[:800]})
            context_parts.append(chunk_text)
        answer = self._generate_answer(question, "\n\n".join(context_parts), session.history)
        session.history.append({"question": question, "answer": answer})
        return {
            "session_id": session_id,
            "source": session.source,
            "question": question,
            "answer": answer,
            "citations": selected,
            "turns": len(session.history),
        }

    def ask_multi(self, session_ids: list[str], question: str, top_k_per_session: int = 3) -> dict:
        if not session_ids:
            raise ValueError("No session IDs provided.")
        collected: list[dict] = []
        for session_id in session_ids:
            try:
                session = self.memory.get(session_id)
            except KeyError:
                logger.warning("Session %s not found, skipping.", session_id)
                continue
            query_vec = self.embedding_service.encode([question]).astype(np.float32)
            scores, ids = session.index.search(query_vec, top_k_per_session)
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0:
                    continue
                collected.append({
                    "session_id": session_id,
                    "source": session.source,
                    "chunk_id": int(idx),
                    "score": float(score),
                    "text": session.chunks[int(idx)],
                })
        if not collected:
            raise ValueError("No retrievable content found across selected sessions.")
        collected.sort(key=lambda item: item["score"], reverse=True)
        top = collected[:10]
        context = "\n\n".join(f"Source: {item['source']}\n{item['text']}" for item in top)
        answer = self._generate_answer(question, context, [])
        return {
            "question": question,
            "answer": answer,
            "citations": [
                {
                    "session_id": item["session_id"],
                    "source": item["source"],
                    "score": round(item["score"], 4),
                    "text": item["text"][:600],
                }
                for item in top
            ],
            "paper_count": len({item["source"] for item in top}),
        }

    def session_info(self, session_id: str) -> dict:
        session = self.memory.get(session_id)
        return {
            "session_id": session.session_id,
            "source": session.source,
            "chunk_count": len(session.chunks),
            "turns": len(session.history),
            "title": session.title,
        }
```
