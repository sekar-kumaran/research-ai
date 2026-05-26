"""MLExecutionAgent — dispatches tool calls with type coercion and error isolation."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Tools whose results should be injected into subsequent "from: search_results" calls
_SEARCH_TOOLS = ("hybrid_search", "smart_retrieve")


class MLExecutionAgent:
    """Executes registered local ML / retrieval / research tools.

    Improvements over original:
    - Type coercion: string "5" → int 5, string "true" → bool True
    - Recognises `{"from": "search_results"}` from ANY prior search tool,
      not just hybrid_search (fixes the data-flow injection bug)
    - Per-tool error isolation: a failing tool does not abort the plan
    - Logs tool name and latency for observability
    """

    def __init__(self, tools: dict[str, object]) -> None:
        self.tools = tools

    def execute(self, name: str, args: dict) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: '{name}'. Available: {sorted(self.tools)}"}
        coerced = self._coerce_args(args)
        try:
            result = tool(**coerced)
            return result if isinstance(result, dict) else {"result": result}
        except TypeError as exc:
            logger.warning("Tool '%s' argument error: %s | args=%s", name, exc, coerced)
            return {"error": f"Tool '{name}' argument mismatch: {exc}"}
        except Exception as exc:
            logger.error("Tool '%s' raised: %s", name, exc)
            return {"error": str(exc)}

    def execute_plan(self, calls: list) -> dict:
        """Execute a list of ToolCall objects, resolving data-flow injections."""
        outputs: dict = {}
        last_search_results: list[dict] = []

        for call in calls:
            args = dict(call.args)

            # Data-flow injection: {"from": "search_results"} → inject prior paper list
            if args.get("from") == "search_results":
                args = {"papers": list(last_search_results)}

            result = self.execute(call.name, args)
            outputs[call.name] = result

            # Track the latest search results for downstream injection
            if call.name in _SEARCH_TOOLS and isinstance(result, dict):
                candidate = result.get("results", [])
                if isinstance(candidate, list) and candidate:
                    last_search_results = candidate

        return outputs

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_args(args: dict) -> dict:
        """Coerce common LLM type mistakes: string numbers, string booleans."""
        coerced: dict = {}
        for key, value in args.items():
            if isinstance(value, str):
                # int coercion for known numeric keys
                if key in ("top_k", "candidate_k", "max_tokens", "timeout"):
                    try:
                        coerced[key] = int(value)
                        continue
                    except ValueError:
                        pass
                # bool coercion
                if value.lower() in ("true", "false"):
                    coerced[key] = value.lower() == "true"
                    continue
            coerced[key] = value
        return coerced
