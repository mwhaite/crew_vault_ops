from __future__ import annotations
from typing import List, Tuple, Type, Dict, Any, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from pathlib import Path
import json, os, yaml, datetime
from .embedder import embed_texts, load_faiss, save_faiss
from .maintenance import run_maintenance

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

    def _bulk_index_vault(self):
        """Indexes the entire vault."""
        chunks, embeddings = [], []
        for md in self.VAULT_PATH.rglob("*.md"):
            with open(md, encoding="utf-8") as f:
                txt = f.read()
            for i, para in enumerate(txt.split("\n\n")):
                if para.strip():
                    chunks.append({"file": str(md), "para": para})
                    embeddings.append(para)
        vecs = embed_texts(embeddings)
        index = load_faiss(vecs.shape[1])
        index.add(vecs)
        save_faiss(index, self.INDEX_FILE)
        self.META_FILE.write_text(json.dumps(chunks, indent=2))

    def _index_file(self, path: Path, overwrite=False):
        """Indexes a single file.

        Args:
            path: The path to the file.
            overwrite: Whether to overwrite the existing index.
        """
        self._ensure_index()
        index, chunks = load_faiss(self.INDEX_FILE), json.loads(self.META_FILE.read_text())
        if overwrite:
            self._remove_from_index(str(path))
        with open(path, encoding="utf-8") as f:
            txt = f.read()
        vecs = embed_texts([txt])
        index.add(vecs)
        chunks.append({"file": str(path), "para": txt})
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

    def _ask(self, question: str, k: int = 5) -> str:
        """Asks a question and returns the answer.

        Args:
            question: The question to ask.
            k: The number of results to return.

        Returns:
            The answer to the question.
        """
        self._ensure_index()
        index, chunks = load_faiss(self.INDEX_FILE), json.loads(self.META_FILE.read_text())
        qvec = embed_texts([question])
        D, I = index.search(qvec, k)
        context = "\n\n".join(chunks[i]["para"] for i in I[0])
        answer = (
            f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
            "Answer (simulated): ...\n"
        )
        return answer

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
        action: str,
        path: str | None = None,
        text: str | None = None,
        maintenance_tasks: List[str] | None = None,
    ) -> str:
        match action:
            case "create": return self._create(path, text or "")
            case "read":   return self._read(self._abs(path))
            case "update": return self._update(path, text or "")
            case "delete": return self._delete(path)
            case "ask":    return self._ask(text or "")
            case "maintenance": return self._maintenance(maintenance_tasks or [])
            case _:        return "Unknown action."

    async def _arun(self, **kwargs):
        return self._run(**kwargs)
