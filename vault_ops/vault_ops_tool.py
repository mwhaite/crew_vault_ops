from __future__ import annotations
from typing import List, Tuple, Type, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from pathlib import Path
import json, numpy as np, re
import mistune
from .embedder import embed_texts, load_faiss, save_faiss
from .maintenance import run_maintenance


class VaultOpsInput(BaseModel):
    action: Literal["create", "read", "update", "delete", "ask", "maintenance"] = Field(
        ..., description="Operation to perform: CRUD, ask (RAG), or maintenance."
    )
    path: Optional[str] = Field(
        None, description="Vault-relative path for CRUD actions (e.g., notes/example.md)."
    )
    text: Optional[str] = Field(
        None,
        description=(
            "Note content for create/update or the question text for ask (RAG retrieval)."
        ),
    )
    maintenance_tasks: Optional[List[str]] = Field(
        None,
        description="Optional list of maintenance task names to run (defaults to all).",
    )

class VaultOpsTool(BaseTool):
    name: str = "vault_ops"
    description: str = (
        "Interact with an Obsidian vault. "
        "Supports RAG Q&A ('ask') plus CRUD and maintenance."
    )
    args_schema: Type[BaseModel] = VaultOpsInput

    def __init__(self, vault_path: str = "./vault"):
        """Initializes the VaultOpsTool.

        Args:
            vault_path: The path to the Obsidian vault.
        """
        super().__init__()
        self.VAULT_PATH = Path(vault_path).resolve()
        self.INDEX_DIR   = self.VAULT_PATH / ".index"
        self.INDEX_FILE  = self.INDEX_DIR / "faiss_index.bin"
        self.META_FILE   = self.INDEX_DIR / "chunks.json"
        self._markdown_parser = mistune.create_markdown(renderer="ast")
        self._tag_pattern = re.compile(r"(?<!\w)#([A-Za-z0-9][\w\-/]*)")
        self._wiki_pattern = re.compile(r"\[\[([^\]]+)\]\]")

    def _note_id(self, path: Path) -> str:
        """Returns a normalized note identifier without extension."""

        rel = path.resolve().relative_to(self.VAULT_PATH)
        return rel.with_suffix("").as_posix()

    def _normalize_link_target(self, target: str) -> str:
        """Normalizes wiki link target text to align with note identifiers."""

        cleaned = target.strip()
        if cleaned.lower().endswith(".md"):
            cleaned = cleaned[:-3]
        return Path(cleaned).with_suffix("").as_posix()

    def _extract_inline_text(self, token: Dict[str, Any]) -> str:
        """Extracts plain text from a mistune token tree."""

        ttype = token.get("type")
        if ttype in {"text", "codespan", "block_code"}:
            return token.get("text", "")
        if ttype == "linebreak":
            return "\n"
        children = token.get("children", [])
        return "".join(self._extract_inline_text(child) for child in children)

    def _build_chunk_text(self, token: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Converts a block token into (content, type) if supported."""

        ttype = token.get("type")
        if ttype == "heading":
            level = token.get("level", 1)
            text = self._extract_inline_text(token)
            return f"{'#' * level} {text}".strip(), ttype
        if ttype == "paragraph":
            return self._extract_inline_text(token).strip(), ttype
        if ttype == "block_code":
            return token.get("text", "").strip(), ttype
        if ttype == "list":
            items = []
            for item in token.get("children", []):
                item_text = self._extract_inline_text(item).strip()
                if item_text:
                    items.append(f"- {item_text}")
            return "\n".join(items), ttype
        if ttype == "block_quote":
            return self._extract_inline_text(token).strip(), ttype
        return None

    def _parse_markdown_chunks(self, path: Path, text: str) -> List[Dict[str, Any]]:
        """Parses markdown into semantic chunks with metadata."""

        tokens = self._markdown_parser(text)
        note_id = self._note_id(path)
        chunks: List[Dict[str, Any]] = []
        for token in tokens:
            chunk_text_type = self._build_chunk_text(token)
            if not chunk_text_type:
                continue
            chunk_text, chunk_type = chunk_text_type
            if not chunk_text:
                continue
            tags = sorted(set(self._tag_pattern.findall(chunk_text)))
            links = sorted(
                {self._normalize_link_target(m) for m in self._wiki_pattern.findall(chunk_text)}
            )
            chunks.append(
                {
                    "file": str(path),
                    "note": note_id,
                    "content": chunk_text,
                    "type": chunk_type,
                    "tags": tags,
                    "links": links,
                }
            )
        return chunks

    def _abs(self, rel: str) -> Path:
        """Converts a relative path to an absolute path and ensures it is within the vault.

        Args:
            rel: The relative path.

        Returns:
            The absolute path.
        """
        p = (self.VAULT_PATH / rel).resolve()
        if not p.is_relative_to(self.VAULT_PATH):
            raise ValueError("Path escapes vault")
        return p

    def _write(self, path: Path, data: str) -> None:
        """Writes data to a file.

        Args:
            path: The path to the file.
            data: The data to write.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)

    def _read(self, path: Path) -> str:
        """Reads data from a file.

        Args:
            path: The path to the file.

        Returns:
            The file content.
        """
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _create(self, rel: str, content: str) -> str:
        """Creates a new file.

        Args:
            rel: The relative path to the file.
            content: The content to write to the file.

        Returns:
            A message indicating the result.
        """
        path = self._abs(rel)
        if path.exists():
            return f"File {rel} already exists."
        self._write(path, content)
        self._index_file(path)
        return f"Created {rel}."

    def _update(self, rel: str, content: str) -> str:
        """Updates an existing file.

        Args:
            rel: The relative path to the file.
            content: The content to write to the file.

        Returns:
            A message indicating the result.
        """
        path = self._abs(rel)
        if not path.exists():
            return f"File {rel} not found."
        self._write(path, content)
        self._index_file(path, overwrite=True)
        return f"Updated {rel}."

    def _delete(self, rel: str) -> str:
        """Deletes a file.

        Args:
            rel: The relative path to the file.

        Returns:
            A message indicating the result.
        """
        path = self._abs(rel)
        if not path.exists():
            return f"{rel} does not exist."
        path.unlink()
        self._remove_from_index(str(path))
        return f"Deleted {rel}."

    def _ensure_index(self):
        """Ensures that a FAISS index exists. If it doesn't, it creates one."""
        self.INDEX_DIR.mkdir(exist_ok=True)
        if not self.INDEX_FILE.exists():
            self._bulk_index_vault()
            return True
        return False

    def _bulk_index_vault(self):
        """Indexes the entire vault."""
        self.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        chunks, embeddings = [], []
        for md in self.VAULT_PATH.rglob("*.md"):
            with open(md, encoding="utf-8") as f:
                txt = f.read()
            file_chunks = self._parse_markdown_chunks(md, txt)
            chunks.extend(file_chunks)
            embeddings.extend([c["content"] for c in file_chunks])
        vecs = embed_texts(embeddings) if embeddings else np.empty((0, 0))
        d = vecs.shape[1] if vecs.size else embed_texts([""]).shape[1]
        index = load_faiss(self.INDEX_FILE, d)
        if vecs.size:
            index.add(vecs)
        save_faiss(index, self.INDEX_FILE)
        self.META_FILE.write_text(json.dumps(chunks, indent=2))

    def _index_file(self, path: Path, overwrite=False):
        """Indexes a single file.

        Args:
            path: The path to the file.
            overwrite: Whether to overwrite the existing index.
        """
        index_created = self._ensure_index()
        if overwrite:
            self._remove_from_index(str(path))
            return
        if index_created:
            return
        with open(path, encoding="utf-8") as f:
            txt = f.read()
        file_chunks = self._parse_markdown_chunks(path, txt)
        if not file_chunks:
            return
        vecs = embed_texts([c["content"] for c in file_chunks])
        index = load_faiss(self.INDEX_FILE, vecs.shape[1])
        chunks = json.loads(self.META_FILE.read_text()) if self.META_FILE.exists() else []
        index.add(vecs)
        chunks.extend(file_chunks)
        save_faiss(index, self.INDEX_FILE)
        self.META_FILE.write_text(json.dumps(chunks, indent=2))

    def _remove_from_index(self, file_path: str):
        """Removes a file from the index.

        Args:
            file_path: The path to the file.
        """
        if self.META_FILE.exists():
            self.META_FILE.unlink()
        if self.INDEX_FILE.exists():
            self.INDEX_FILE.unlink()
        self._bulk_index_vault()

    def _build_link_maps(self, chunks: List[Dict[str, Any]]):
        """Builds outgoing and backlink maps from chunk metadata."""

        outgoing: Dict[str, set[str]] = {}
        backlinks: Dict[str, set[str]] = {}
        for chunk in chunks:
            note = chunk.get("note") or self._note_id(Path(chunk["file"]))
            links = chunk.get("links", [])
            if note not in outgoing:
                outgoing[note] = set()
            for link in links:
                normalized = self._normalize_link_target(link)
                outgoing[note].add(normalized)
                backlinks.setdefault(normalized, set()).add(note)
        return outgoing, backlinks

    def _ask(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Asks a question and returns the answer.

        Args:
            question: The question to ask.
            k: The number of results to return.

        Returns:
            A structured response containing the synthesized answer, link trail, and tag cloud.
        """
        self._ensure_index()
        qvec = embed_texts([question])
        index = load_faiss(self.INDEX_FILE, qvec.shape[1])
        chunks = json.loads(self.META_FILE.read_text()) if self.META_FILE.exists() else []
        D, I = index.search(qvec, k)
        top_chunks = [chunks[i] for i in I[0] if 0 <= i < len(chunks)]
        outgoing, backlinks = self._build_link_maps(chunks)

        visited_notes = []
        visited_set = set()
        context_blocks = []
        seen_contents = set()
        all_tags: List[str] = []

        def add_note_chunks(note_id: str, source: str):
            if note_id in visited_set:
                return
            visited_set.add(note_id)
            visited_notes.append({"note": note_id, "via": source})
            for ch in chunks:
                if (ch.get("note") or self._note_id(Path(ch["file"]))) == note_id:
                    content = ch.get("content") or ch.get("para", "")
                    if content and content not in seen_contents:
                        seen_contents.add(content)
                        context_blocks.append(content)
                        all_tags.extend(ch.get("tags", []))

        for ch in top_chunks:
            note_id = ch.get("note") or self._note_id(Path(ch["file"]))
            add_note_chunks(note_id, "retrieval")

        expansion_targets = set()
        for note in list(visited_set):
            expansion_targets.update(outgoing.get(note, set()))
            expansion_targets.update(backlinks.get(note, set()))

        for note in sorted(expansion_targets):
            add_note_chunks(note, "link")

        tag_cloud: List[Dict[str, Any]] = []
        if all_tags:
            counts: Dict[str, int] = {}
            for tag in all_tags:
                counts[tag] = counts.get(tag, 0) + 1
            tag_cloud = [
                {"tag": t, "count": counts[t]} for t in sorted(counts, key=counts.get, reverse=True)
            ]

        context = "\n\n".join(context_blocks)
        answer_text = (
            f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
            "Answer (simulated): ..."
        )
        return {
            "question": question,
            "answer": answer_text,
            "context": context,
            "link_trail": visited_notes,
            "tag_cloud": tag_cloud,
        }

    def _maintenance(self, tasks: List[str] | None) -> str:
        """Runs maintenance tasks.

        Args:
            tasks: A list of maintenance tasks to run.

        Returns:
            A report of the maintenance tasks.
        """
        report = run_maintenance(self.VAULT_PATH, tasks)
        return report

    def _run(
        self,
        action: Literal["create", "read", "update", "delete", "ask", "maintenance"],
        path: Optional[str] = None,
        text: Optional[str] = None,
        maintenance_tasks: Optional[List[str]] = None,
    ) -> str:
        match action:
            case "create": return self._create(path, text or "")
            case "read":   return self._read(self._abs(path))
            case "update": return self._update(path, text or "")
            case "delete": return self._delete(path)
            case "ask":    return self._ask(text or "")
            case "maintenance": return self._maintenance(maintenance_tasks or [])
            case _:        return "Unknown action."

    async def _arun(
        self,
        action: Literal["create", "read", "update", "delete", "ask", "maintenance"],
        path: Optional[str] = None,
        text: Optional[str] = None,
        maintenance_tasks: Optional[List[str]] = None,
    ):
        return self._run(action, path=path, text=text, maintenance_tasks=maintenance_tasks)
