from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from uuid import uuid4

from .cloud_llm import CloudLLMClient
from .paper_chat import PaperChatService
from .preprocess import build_full_text, clean_text

logger = logging.getLogger(__name__)


@dataclass
class AssistantAgents:
    classifier: object | None
    vectorizer: object | None
    rag_assistant: object | None
    summarizer: object | None
    paper_chat: PaperChatService | None
    _cloud_client: CloudLLMClient | None = field(default=None, init=False, repr=False)

    _ALLOWED_INTENTS = {"ask", "search", "classify", "summarize", "paper_chat"}

    MEDIATOR_SYSTEM = (
        "You are a routing mediator for a hybrid research assistant. "
        "Return JSON only with keys: intent, normalized_query, top_k, title, abstract, text, reason. "
        "intent must be one of ask, search, classify, summarize, paper_chat. "
        "Do not include markdown or explanations outside JSON."
    )

    SYNTHESIS_SYSTEM = (
        "You are a user-friendly research assistant. Convert model execution results into a concise, "
        "grounded final response. Use only provided evidence. If evidence is weak, say so clearly."
    )

    def _ensure_cloud(self) -> CloudLLMClient | None:
        if self._cloud_client is not None:
            return self._cloud_client
        try:
            self._cloud_client = CloudLLMClient()
            return self._cloud_client
        except Exception as exc:
            logger.info(f"Cloud mediator unavailable, using fallback routing: {exc}")
            return None

    @staticmethod
    def _clamp_top_k(value: int) -> int:
        return max(1, min(20, int(value)))

    @staticmethod
    def _safe_json_parse(text: str) -> dict | None:
        if not text:
            return None
        payload = text.strip()
        if "```" in payload:
            payload = payload.replace("```json", "").replace("```", "").strip()
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            parsed = json.loads(payload[start : end + 1])
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _derive_title_abstract(query: str, title: str | None, abstract: str | None) -> tuple[str, str]:
        if title and abstract:
            return title, abstract
        if title and not abstract:
            return title, title
        q = (query or "").strip()
        parts = [p.strip() for p in q.splitlines() if p.strip()]
        if len(parts) >= 2:
            return parts[0], " ".join(parts[1:])
        fallback = q or "Untitled"
        return fallback, fallback

    def _heuristic_plan(
        self,
        mode: str,
        query: str,
        top_k: int,
        title: str | None,
        abstract: str | None,
        text: str | None,
        session_id: str | None,
    ) -> dict:
        q = (query or "").strip()
        mode_hint = (mode or "auto").strip().lower()
        if mode_hint in self._ALLOWED_INTENTS:
            intent = mode_hint
        else:
            lowered = q.lower()
            if session_id:
                intent = "paper_chat"
            elif any(k in lowered for k in ("classify", "category", "categorize")):
                intent = "classify"
            elif any(k in lowered for k in ("summarize", "summary", "tldr")):
                intent = "summarize"
            elif any(k in lowered for k in ("search", "find", "papers on", "look up")):
                intent = "search"
            else:
                intent = "ask"

        title_val, abstract_val = self._derive_title_abstract(q, title, abstract)
        return {
            "intent": intent,
            "normalized_query": q,
            "top_k": self._clamp_top_k(top_k),
            "title": title_val,
            "abstract": abstract_val,
            "text": text or q,
            "session_id": session_id,
            "reason": "heuristic_fallback",
            "used_fallback": True,
        }

    def _build_mediator_prompt(
        self,
        mode: str,
        query: str,
        top_k: int,
        title: str | None,
        abstract: str | None,
        text: str | None,
        session_id: str | None,
    ) -> str:
        return (
            "Decide how to route this user request to tools/models.\n"
            f"mode_hint: {mode}\n"
            f"query: {query}\n"
            f"title: {title or ''}\n"
            f"abstract: {abstract or ''}\n"
            f"text: {text or ''}\n"
            f"session_id: {session_id or ''}\n"
            f"top_k: {self._clamp_top_k(top_k)}\n\n"
            "Return strict JSON only with this shape:\n"
            "{\n"
            '  "intent": "ask|search|classify|summarize|paper_chat",\n'
            '  "normalized_query": "string",\n'
            '  "top_k": 5,\n'
            '  "title": "string or empty",\n'
            '  "abstract": "string or empty",\n'
            '  "text": "string or empty",\n'
            '  "reason": "short reason"\n'
            "}"
        )

    def _mediator_plan(
        self,
        mode: str,
        query: str,
        top_k: int,
        title: str | None,
        abstract: str | None,
        text: str | None,
        session_id: str | None,
    ) -> dict:
        fallback = self._heuristic_plan(mode, query, top_k, title, abstract, text, session_id)
        cloud = self._ensure_cloud()
        if cloud is None:
            return fallback

        prompt = self._build_mediator_prompt(mode, query, top_k, title, abstract, text, session_id)
        try:
            raw = cloud.generate(prompt=prompt, max_tokens=220, system=self.MEDIATOR_SYSTEM)
            parsed = self._safe_json_parse(raw)
            if not parsed:
                return fallback

            intent = str(parsed.get("intent", "")).strip().lower()
            if intent not in self._ALLOWED_INTENTS:
                intent = fallback["intent"]

            title_val, abstract_val = self._derive_title_abstract(
                str(parsed.get("normalized_query") or query or "").strip(),
                str(parsed.get("title") or "").strip() or title,
                str(parsed.get("abstract") or "").strip() or abstract,
            )

            return {
                "intent": intent,
                "normalized_query": str(parsed.get("normalized_query") or query or "").strip(),
                "top_k": self._clamp_top_k(parsed.get("top_k") or top_k),
                "title": title_val,
                "abstract": abstract_val,
                "text": str(parsed.get("text") or text or query or "").strip(),
                "session_id": session_id,
                "reason": str(parsed.get("reason") or "mediator_route").strip(),
                "used_fallback": False,
            }
        except Exception as exc:
            logger.warning(f"Mediator planning failed, fallback applied: {exc}")
            return fallback

    @staticmethod
    def _executor_text(intent: str, output: dict) -> str:
        if not isinstance(output, dict):
            return str(output)
        if output.get("error"):
            return str(output["error"])
        if intent == "classify":
            cat = output.get("predicted_category", "unknown")
            conf = output.get("confidence", {})
            if conf:
                top = next(iter(conf.items()))
                return f"Predicted category: {cat}. Top confidence: {top[0]}={top[1]}."
            return f"Predicted category: {cat}."
        if intent == "search":
            count = output.get("count", 0)
            return f"Found {count} relevant papers."
        if intent == "summarize":
            return str(output.get("summary", ""))
        if intent == "paper_chat":
            return str(output.get("answer", ""))
        for key in ("final_answer", "answer", "summary"):
            val = output.get(key)
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, dict):
                nested = val.get("answer") or val.get("final_answer")
                if isinstance(nested, str) and nested.strip():
                    return nested
        try:
            return json.dumps(output, ensure_ascii=False)
        except Exception:
            return str(output)

    def _execute_mediator_plan(self, plan: dict) -> dict:
        intent = plan["intent"]
        if intent == "classify":
            return self.classify(plan.get("title", ""), plan.get("abstract", ""))
        if intent == "search":
            return self.search(plan.get("normalized_query", ""), top_k=plan.get("top_k", 5))
        if intent == "summarize":
            return self.summarize(plan.get("text", ""))
        if intent == "paper_chat":
            if not self.paper_chat:
                return {"error": "Paper chat service unavailable."}
            sid = str(plan.get("session_id") or "").strip()
            if not sid:
                return {"error": "session_id required for paper_chat intent."}
            try:
                return self.paper_chat.ask(
                    session_id=sid,
                    question=plan.get("normalized_query", ""),
                    top_k=plan.get("top_k", 5),
                )
            except Exception as exc:
                return {"error": f"Paper chat failed: {exc}"}
        return self.ask(plan.get("normalized_query", ""), top_k=plan.get("top_k", 5))

    def _synthesize_final_answer(self, user_query: str, plan: dict, executor_output: dict) -> str:
        if isinstance(executor_output, dict) and executor_output.get("error"):
            return f"I could not complete this request: {executor_output['error']}"

        cloud = self._ensure_cloud()
        fallback_text = self._executor_text(plan.get("intent", "ask"), executor_output)
        if cloud is None:
            return fallback_text

        synthesis_prompt = (
            f"User query:\n{user_query}\n\n"
            f"Mediator intent: {plan.get('intent')}\n"
            f"Mediator reason: {plan.get('reason')}\n\n"
            "Model executor output (JSON):\n"
            f"{json.dumps(executor_output, ensure_ascii=False)[:12000]}\n\n"
            "Write a final helpful response for the user. Keep it concise, grounded, and clear."
        )
        try:
            final = cloud.generate(prompt=synthesis_prompt, max_tokens=420, system=self.SYNTHESIS_SYSTEM)
            final = (final or "").strip()
            return final or fallback_text
        except Exception as exc:
            logger.warning(f"Synthesis failed, using fallback text: {exc}")
            return fallback_text

    def mediated_run(
        self,
        mode: str,
        query: str,
        title: str | None = None,
        abstract: str | None = None,
        top_k: int = 5,
        text: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        started = time.perf_counter()
        request_id = str(uuid4())

        plan = self._mediator_plan(
            mode=mode,
            query=query,
            top_k=top_k,
            title=title,
            abstract=abstract,
            text=text,
            session_id=session_id,
        )
        executor_output = self._execute_mediator_plan(plan)
        final_answer = self._synthesize_final_answer(query, plan, executor_output)

        return {
            "request_id": request_id,
            "mode": plan.get("intent", "ask"),
            "mediator": {
                "reason": plan.get("reason", ""),
                "used_fallback": bool(plan.get("used_fallback", False)),
            },
            "executor_output": executor_output,
            "final_answer": final_answer,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def classify(self, title: str, abstract: str) -> dict:
        if self.classifier is None or self.vectorizer is None:
            return {"error": "Classifier artifacts missing. Run training pipeline first."}
        text = clean_text(build_full_text(title, abstract))
        x = self.vectorizer.transform([text])
        pred = self.classifier.predict(x)[0]
        try:
            proba = self.classifier.predict_proba(x)[0]
            classes = self.classifier.classes_
            confidence = {str(c): round(float(p), 4) for c, p in sorted(zip(classes, proba), key=lambda x: -x[1])[:5]}
        except Exception:
            confidence = {}
        return {"predicted_category": str(pred), "confidence": confidence}

    def search(self, query: str, top_k: int = 5) -> dict:
        if self.rag_assistant is None:
            return {"error": "Similarity index missing. Build embeddings first."}
        docs = self.rag_assistant.retrieve(query, top_k=top_k)
        return {"results": [d.to_dict() for d in docs], "count": len(docs)}

    def summarize(self, text: str) -> dict:
        if self.summarizer is None:
            return {"error": "Summarizer unavailable. Check LLM_BACKEND configuration."}
        return {"summary": self.summarizer.summarize(text)}

    def ask(self, query: str, top_k: int = 5) -> dict:
        if self.rag_assistant is None:
            return {"error": "RAG components missing. Build similarity artifacts first."}
        return self._answer_with_papers(query=query, top_k=top_k)

    @staticmethod
    def _synthesize_from_results(query: str, results: list[dict], max_items: int = 5) -> str:
        if not results:
            return "No matching papers found in the index for your query."
        lines = [f"Here are the top papers related to: **{query}**\n"]
        for idx, item in enumerate(results[:max_items], 1):
            title = str(item.get("title", "Untitled")).strip()
            paper_id = str(item.get("paper_id", "")).strip()
            abstract = str(item.get("abstract", "")).strip().replace("\n", " ")
            brief = " ".join(abstract.split())[:280]
            if len(abstract) > 280:
                brief += "…"
            year = item.get("year", "")
            category = item.get("category", "")
            url = f"https://arxiv.org/abs/{paper_id}" if paper_id else ""
            lines.append(f"**{idx}. {title}** ({year}) [{category}]")
            if url:
                lines.append(f"   🔗 {url}")
            if brief:
                lines.append(f"   {brief}\n")
        lines.append("\n*Ask me to summarize any of these papers for more detail.*")
        return "\n".join(lines)

    @staticmethod
    def _looks_low_quality(text: str) -> bool:
        t = (text or "").strip()
        if not t or len(t.split()) < 12:
            return True
        if t.count("|") >= 2 or t.count("[") >= 3:
            return True
        lowered = t.lower()
        filler = ("university", "department", "campus", "institute")
        if any(w in lowered for w in filler) and lowered.count(".") <= 1:
            return True
        return False

    def _answer_with_papers(self, query: str, top_k: int = 5, download_top_n: int = 2) -> dict:
        search = self.search(query, top_k=top_k)
        if "error" in search:
            return search

        results = search.get("results", [])

        # Try downloading full papers for deep QA
        if self.paper_chat:
            selected_ids: list[str] = []
            for item in results:
                nid = self.paper_chat.normalize_arxiv_id(str(item.get("paper_id", "")))
                if nid and nid not in selected_ids:
                    selected_ids.append(nid)
                if len(selected_ids) >= download_top_n:
                    break

            sessions = []
            load_events = []
            for arxiv_id in selected_ids:
                try:
                    meta = self.paper_chat.create_or_get_session_from_arxiv_id(arxiv_id)
                    sessions.append(meta["session_id"])
                    load_events.append({
                        "arxiv_id": arxiv_id,
                        "session_id": meta["session_id"],
                        "source": meta.get("source", f"arxiv:{arxiv_id}"),
                        "cached": bool(meta.get("cached", False)),
                    })
                except Exception as exc:
                    load_events.append({"arxiv_id": arxiv_id, "error": str(exc)})
                    logger.warning(f"Could not load paper {arxiv_id}: {exc}")

            if sessions:
                try:
                    paper_answer = self.paper_chat.ask_multi(session_ids=sessions, question=query, top_k_per_session=3)
                    candidate = str(paper_answer.get("answer", "")).strip()
                    if not self._looks_low_quality(candidate):
                        return {
                            "mode": "ask",
                            "strategy": "retrieve_then_deep_qa",
                            "search": search,
                            "downloaded_papers": load_events,
                            "paper_answer": paper_answer,
                            "final_answer": candidate,
                        }
                except Exception as exc:
                    logger.warning(f"Multi-paper QA failed: {exc}")

        # Fallback to metadata RAG
        if self.rag_assistant:
            try:
                rag_out = self.rag_assistant.answer(query, top_k=top_k)
                answer_text = str(rag_out.get("answer", "")).strip()
                if not self._looks_low_quality(answer_text):
                    return {
                        "mode": "ask",
                        "strategy": "metadata_rag",
                        "search": search,
                        "answer": rag_out,
                        "final_answer": answer_text,
                    }
            except Exception as exc:
                logger.warning(f"RAG answer failed: {exc}")

        # Final fallback: structured synthesis from metadata
        return {
            "mode": "ask",
            "strategy": "metadata_synthesis",
            "search": search,
            "final_answer": self._synthesize_from_results(query, results),
        }

    def run(
        self,
        mode: str,
        query: str,
        title: str | None = None,
        abstract: str | None = None,
        top_k: int = 5,
        text: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        return self.mediated_run(
            mode=mode,
            query=query,
            title=title,
            abstract=abstract,
            top_k=top_k,
            text=text,
            session_id=session_id,
        )
