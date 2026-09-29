"""OpenAI-native RAG: Files + Vector Store + ``file_search``.

No custom vector DB, no local chunking/embeddings pipeline — this is a
thin wrapper around three OpenAI-hosted primitives:

1. ``client.files`` — upload the raw PDF/MD/TXT.
2. ``client.vector_stores`` — an OpenAI-managed vector index the uploaded
   file is attached to.
3. The ``file_search`` hosted tool on the Responses API — semantic +
   keyword retrieval over that vector store, called with a plain
   (non-JSON, non-agent) prompt so retrieval never has to share a call
   with the strict JSON contract the primary agent depends on.

The grounding call intentionally asks for a short synthesized answer
*and* returns the individual retrieved snippets (via
``include=["file_search_call.results"]``) so the UI can show exactly
what was retrieved, not just the model's paraphrase of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from . import openai_client

_GROUNDING_INSTRUCTIONS = (
    "Answer the question using ONLY the attached file(s), retrieved via file "
    "search. Be concise and factual (a few sentences). If the attached file(s) "
    "don't contain anything relevant to the question, say plainly that nothing "
    "relevant was found — do not answer from general knowledge instead."
)


@dataclass
class RetrievedChunk:
    file_name: str
    text: str
    score: Optional[float] = None


class OpenAIRag:
    """Manages one OpenAI vector store for the current session."""

    def __init__(self, client, *, vector_store_name: str = "agentplant-rag-store", model: Optional[str] = None):
        self.client = client
        self.model = model or openai_client.DEFAULT_MODEL
        self.vector_store_name = vector_store_name
        self.vector_store_id: Optional[str] = None
        self.file_names: List[str] = []
        # `vector_stores` moved out of `beta` in newer SDKs; support both.
        self._vs = getattr(client, "vector_stores", None) or client.beta.vector_stores

    @property
    def has_files(self) -> bool:
        return bool(self.file_names)

    def ensure_store(self) -> str:
        if self.vector_store_id:
            return self.vector_store_id
        vs = self._vs.create(name=self.vector_store_name)
        self.vector_store_id = vs.id
        return self.vector_store_id

    def add_file(self, file_path: str, display_name: Optional[str] = None) -> None:
        """Upload a file and attach it to this session's vector store."""
        vs_id = self.ensure_store()
        fb = getattr(self._vs, "file_batches", None)
        with open(file_path, "rb") as fh:
            if fb is not None and hasattr(fb, "upload_and_poll"):
                fb.upload_and_poll(vector_store_id=vs_id, files=[fh])
            else:
                # Older SDKs: upload the file, then attach it explicitly.
                file_obj = self.client.files.create(file=fh, purpose="assistants")
                self._vs.files.create(vector_store_id=vs_id, file_id=file_obj.id)
        self.file_names.append(display_name or file_path.rsplit("/", 1)[-1])

    def retrieve(self, query: str, *, max_results: int = 5) -> Tuple[str, List[RetrievedChunk]]:
        """Ground ``query`` against the vector store.

        Returns ``(summary_text, chunks)`` — ``summary_text`` is a short
        synthesized answer for prompt injection, ``chunks`` is what was
        actually retrieved, for the "show what was retrieved" UI.
        """
        vs_id = self.ensure_store()
        resp = openai_client.responses_call(
            self.client,
            kind="rag_retrieve",
            model=self.model,
            instructions=_GROUNDING_INSTRUCTIONS,
            input_text=query,
            tools=[{"type": "file_search", "vector_store_ids": [vs_id], "max_num_results": max_results}],
            include=["file_search_call.results"],
        )
        summary = resp.output_text or ""
        chunks = self._parse_results(resp)
        return summary, chunks

    @staticmethod
    def _parse_results(resp: Any) -> List[RetrievedChunk]:
        chunks: List[RetrievedChunk] = []
        for item in getattr(resp, "output", None) or []:
            if getattr(item, "type", None) != "file_search_call":
                continue
            for r in getattr(item, "results", None) or []:
                file_name = getattr(r, "filename", None) or getattr(r, "file_id", None) or "file"
                text = getattr(r, "text", None) or ""
                score = getattr(r, "score", None)
                chunks.append(RetrievedChunk(file_name=file_name, text=text[:600], score=score))
        return chunks
