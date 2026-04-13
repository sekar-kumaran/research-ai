from __future__ import annotations

from dataclasses import dataclass

from langchain_classic.agents import AgentType, Tool, initialize_agent

from .langchain_runtime import get_langchain_llm
from .preprocess import build_full_text, clean_text
from .paper_chat import PaperChatService


@dataclass
class AssistantAgents:
    classifier: object | None
    vectorizer: object | None
    rag_assistant: object | None
    summarizer: object | None
    paper_chat: PaperChatService | None

    def _build_langchain_agent(self):
        llm = get_langchain_llm()

        tools = [
            Tool(
                name="classify_paper",
                func=lambda q: str(self.classify(q, q)),
                description="Use this for category prediction of a paper text. Input should be paper text or title+abstract.",
            ),
            Tool(
                name="search_papers",
                func=lambda q: str(self.search(q, top_k=5)),
                description="Use this to retrieve relevant papers for a query.",
            ),
            Tool(
                name="summarize_text",
                func=lambda q: str(self.summarize(q)),
                description="Use this to summarize a long abstract or paper snippet.",
            ),
            Tool(
                name="rag_answer",
                func=lambda q: str(self.ask(q, top_k=5)),
                description="Use this to answer a research question with retrieved context.",
            ),
        ]

        return initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
        )

    def classify(self, title: str, abstract: str) -> dict:
        if self.classifier is None or self.vectorizer is None:
            return {"error": "Classifier artifacts missing. Run training first."}
        text = clean_text(build_full_text(title, abstract))
        x = self.vectorizer.transform([text])
        pred = self.classifier.predict(x)[0]
        return {"predicted_category": str(pred)}

    def search(self, query: str, top_k: int = 5) -> dict:
        if self.rag_assistant is None:
            return {"error": "Similarity index missing. Build embeddings first."}
        docs = self.rag_assistant.retrieve(query, top_k=top_k)
        return {"results": [d.__dict__ for d in docs]}

    def summarize(self, text: str) -> dict:
        if self.summarizer is None:
            return {"error": "Summarizer is unavailable in current environment."}
        return {"summary": self.summarizer.summarize(text)}

    def ask(self, query: str, top_k: int = 5) -> dict:
        if self.rag_assistant is None:
            return {"error": "RAG components missing. Build similarity artifacts first."}
        return self.answer_with_downloaded_papers(query=query, top_k=top_k)

    @staticmethod
    def _user_text_from_answer_payload(payload: dict) -> str | None:
        if not isinstance(payload, dict):
            return None

        final_answer = payload.get("final_answer")
        if isinstance(final_answer, str) and final_answer.strip():
            return final_answer.strip()

        # Preferred: direct answer string.
        direct = payload.get("answer")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        if isinstance(direct, dict):
            nested = direct.get("answer")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()

        # Alternate: nested paper QA answer.
        paper_answer = payload.get("paper_answer")
        if isinstance(paper_answer, dict):
            candidate = paper_answer.get("answer")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        return None

    @staticmethod
    def _looks_low_quality_answer(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        if len(t.split()) < 12:
            return True

        lowered = t.lower()
        if t.count("|") >= 2 or t.count("[") >= 2:
            return True

        affiliation_terms = ("university", "department", "campus", "institute")
        if any(term in lowered for term in affiliation_terms) and lowered.count(".") <= 1:
            return True

        return False

    @staticmethod
    def _synthesize_from_search_results(query: str, results: list[dict], max_items: int = 3) -> str:
        if not results:
            return "I could not find strong matching papers for your query."

        lines = [f"Here are top papers related to: {query}", ""]
        for idx, item in enumerate(results[:max_items], start=1):
            title = str(item.get("title", "Untitled")).strip()
            paper_id = str(item.get("paper_id", "unknown")).strip()
            abstract = str(item.get("abstract", "")).strip().replace("\n", " ")
            abstract = " ".join(abstract.split())
            brief = abstract[:220].rstrip()
            if brief and len(abstract) > 220:
                brief += "..."

            lines.append(f"{idx}. {title} ({paper_id})")
            if brief:
                lines.append(f"   - {brief}")

        lines.append("")
        lines.append("Ask me to summarize any one of these papers in detail.")
        return "\n".join(lines)

    def answer_with_downloaded_papers(self, query: str, top_k: int = 5, download_top_n: int = 2) -> dict:
        if self.rag_assistant is None:
            return {"error": "RAG components missing. Build similarity artifacts first."}

        search = self.search(query, top_k=top_k)
        if "error" in search:
            return search

        if self.paper_chat is None:
            rag_answer = self.rag_assistant.answer(query, top_k=top_k)
            return {
                "mode": "ask",
                "strategy": "metadata_rag_fallback",
                "search": search,
                "answer": rag_answer,
                "final_answer": self._user_text_from_answer_payload({"answer": rag_answer})
                or self._synthesize_from_search_results(query, search.get("results", [])),
            }

        results = search.get("results", [])
        selected_ids: list[str] = []
        for item in results:
            normalized = self.paper_chat.normalize_arxiv_id(str(item.get("paper_id", "")))
            if normalized and normalized not in selected_ids:
                selected_ids.append(normalized)
            if len(selected_ids) >= download_top_n:
                break

        sessions = []
        load_events = []
        for arxiv_id in selected_ids:
            try:
                meta = self.paper_chat.create_or_get_session_from_arxiv_id(arxiv_id)
                sessions.append(meta["session_id"])
                load_events.append(
                    {
                        "arxiv_id": arxiv_id,
                        "session_id": meta["session_id"],
                        "source": meta.get("source", f"arxiv:{arxiv_id}"),
                        "cached": bool(meta.get("cached", False)),
                    }
                )
            except Exception as exc:
                load_events.append({"arxiv_id": arxiv_id, "error": str(exc)})

        if sessions:
            try:
                paper_answer = self.paper_chat.ask_multi(session_ids=sessions, question=query, top_k_per_session=3)
                candidate = str(paper_answer.get("answer", "")).strip()
                if self._looks_low_quality_answer(candidate):
                    final_answer = self._synthesize_from_search_results(query, results)
                    strategy = "retrieve_then_download_then_synthesized_from_metadata"
                else:
                    final_answer = candidate
                    strategy = "retrieve_then_download_then_answer"

                return {
                    "mode": "ask",
                    "strategy": strategy,
                    "search": search,
                    "downloaded_papers": load_events,
                    "paper_answer": paper_answer,
                    "final_answer": final_answer,
                }
            except Exception as exc:
                rag_answer = self.rag_assistant.answer(query, top_k=top_k)
                return {
                    "mode": "ask",
                    "strategy": "partial_failure_fallback_to_metadata_rag",
                    "search": search,
                    "downloaded_papers": load_events,
                    "paper_qa_error": str(exc),
                    "answer": rag_answer,
                    "final_answer": self._user_text_from_answer_payload({"answer": rag_answer})
                    or self._synthesize_from_search_results(query, results),
                }

        rag_answer = self.rag_assistant.answer(query, top_k=top_k)
        return {
            "mode": "ask",
            "strategy": "no_downloadable_paper_id_fallback_to_metadata_rag",
            "search": search,
            "downloaded_papers": load_events,
            "answer": rag_answer,
            "final_answer": self._user_text_from_answer_payload({"answer": rag_answer})
            or self._synthesize_from_search_results(query, results),
        }

    def run(self, mode: str, query: str, title: str | None = None, abstract: str | None = None, top_k: int = 5) -> dict:
        if mode == "classify":
            return self.classify(title or query, abstract or query)
        if mode == "search":
            return self.search(query, top_k=top_k)
        if mode == "summarize":
            return self.summarize(query)
        if mode == "ask":
            return self.ask(query, top_k=top_k)

        # auto mode is now driven by a LangChain agent over available tools.
        agent = self._build_langchain_agent()
        final_text = agent.run(
            "You are an AI research assistant. Choose the best tool(s) and answer the user query. "
            f"User query: {query}"
        )

        search_out = self.search(query, top_k=top_k)
        answer_out = self.answer_with_downloaded_papers(query, top_k=top_k)
        answer_text = self._user_text_from_answer_payload(answer_out)

        if isinstance(answer_text, str) and answer_text.strip():
            final_text = answer_text

        if isinstance(final_text, str) and "iteration limit" in final_text.lower():
            final_text = (
                self._user_text_from_answer_payload(answer_out)
                or "I could not complete tool reasoning in time, but here is the best available answer from retrieved research context."
            )

        return {
            "mode": "auto",
            "agent_framework": "langchain",
            "agent_output": final_text,
            "search": search_out,
            "answer": answer_out,
        }
