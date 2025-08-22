from __future__ import annotations
from typing import List, Tuple, Type, Dict, Any, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from pathlib import Path
import json, os, yaml, datetime
from .embedder import embed_texts, load_faiss, save_faiss
from .maintenance import run_maintenance

VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT", "./vault")).resolve()
INDEX_DIR   = Path("./vector_store").resolve()
INDEX_FILE  = INDEX_DIR / "faiss_index.bin"
META_FILE   = INDEX_DIR / "chunks.json"

class VaultOpsInput(BaseModel):
    action: str = Field(
        ...,
        description=("What to do. Options: 'ask', 'create', 'read', 'update', 'delete', 'maintenance'"),
    )
    path: Optional[str] = Field(
        None,
        description="Relative path (inside vault) for CRUD actions"
    )
    text: Optional[str] = Field(
        None,
        description="Markdown content for create/update, or question for 'ask'"
    )
    maintenance_tasks: Optional[List[str]] = Field(
        None,
        description="Subset of maintenance jobs to run; leave blank for all"
    )

class VaultOpsTool(BaseTool):
    name: str = "vault_ops"
    description: str = (
        "Interact with an Obsidian vault. "
        "Supports RAG Q&A ('ask') plus CRUD and maintenance."
    )
    args_schema: Type[BaseModel] = VaultOpsInput

    def _abs(self, rel: str) -> Path:
        p = (VAULT_PATH / rel).resolve()
        if not p.is_relative_to(VAULT_PATH):
            raise ValueError("Path escapes vault")
        return p

    def _write(self, path: Path, data: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)

    def _read(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _create(self, rel: str, content: str) -> str:
        path = self._abs(rel)
        if path.exists():
            return f"File {rel} already exists."
        self._write(path, content)
        self._index_file(path)
        return f"Created {rel}."

    def _update(self, rel: str, content: str) -> str:
        path = self._abs(rel)
        if not path.exists():
            return f"File {rel} not found."
        self._write(path, content)
        self._index_file(path, overwrite=True)
        return f"Updated {rel}."

    def _delete(self, rel: str) -> str:
        path = self._abs(rel)
        if not path.exists():
            return f"{rel} does not exist."
        path.unlink()
        self._remove_from_index(str(path))
        return f"Deleted {rel}."

    def _ensure_index(self):
        INDEX_DIR.mkdir(exist_ok=True)
        if not INDEX_FILE.exists():
            self._bulk_index_vault()

    def _bulk_index_vault(self):
        chunks, embeddings = [], []
        for md in VAULT_PATH.rglob("*.md"):
            with open(md, encoding="utf-8") as f:
                txt = f.read()
            for i, para in enumerate(txt.split("\n\n")):
                if para.strip():
                    chunks.append({"file": str(md), "para": para})
                    embeddings.append(para)
        vecs = embed_texts(embeddings)
        index = load_faiss(vecs.shape[1])
        index.add(vecs)
        save_faiss(index, INDEX_FILE)
        META_FILE.write_text(json.dumps(chunks, indent=2))

    def _index_file(self, path: Path, overwrite=False):
        self._ensure_index()
        index, chunks = load_faiss(INDEX_FILE), json.loads(META_FILE.read_text())
        if overwrite:
            self._remove_from_index(str(path))
        with open(path, encoding="utf-8") as f:
            txt = f.read()
        vecs = embed_texts([txt])
        index.add(vecs)
        chunks.append({"file": str(path), "para": txt})
        save_faiss(index, INDEX_FILE)
        META_FILE.write_text(json.dumps(chunks, indent=2))

    def _remove_from_index(self, file_path: str):
        if META_FILE.exists():
            META_FILE.unlink()
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()
        self._bulk_index_vault()

    def _ask(self, question: str, k: int = 5) -> str:
        self._ensure_index()
        index, chunks = load_faiss(INDEX_FILE), json.loads(META_FILE.read_text())
        qvec = embed_texts([question])
        D, I = index.search(qvec, k)
        context = "\n\n".join(chunks[i]["para"] for i in I[0])
        answer = (
            f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
            "Answer (simulated): ...\n"
        )
        return answer

    def _maintenance(self, tasks: List[str] | None) -> str:
        report = run_maintenance(VAULT_PATH, tasks)
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
