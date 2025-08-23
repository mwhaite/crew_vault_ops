from pathlib import Path
import re

def run_maintenance(vault: Path, tasks: list[str]):
    log = []
    if not tasks or "empty" in tasks:
        empties = [p for p in vault.rglob("*.md") if p.stat().st_size < 5]
        for p in empties: p.unlink()
        log.append(f"Removed {len(empties)} empty notes.")
    if not tasks or "dangling" in tasks:
        md_files = {str(p) for p in vault.rglob("*.md")}
        pattern = re.compile(r"\[\[([^\]]+)\]\]")
        dangling = []
        for p in vault.rglob("*.md"):
            for target in pattern.findall(p.read_text(encoding="utf-8")):
                tgt_path = vault / (target + ".md")
                if not tgt_path.exists():
                    dangling.append((p, target))
        log.append(f"Found {len(dangling)} dangling links.")
    if not tasks or "orphans" in tasks:
        refs = set()
        for p in vault.rglob("*.md"):
            for m in re.findall(r"!\\[\\]\((.*?)\")", p.read_text()):
                refs.add((vault / m).resolve())
        orphans = [img for img in vault.rglob("*.*") if img.suffix in {".png",".jpg",".pdf"} and img not in refs]
        for o in orphans: o.unlink()
        log.append(f"Deleted {len(orphans)} orphan attachments.")
    if not tasks or "backlinks" in tasks:
        backlinks = {}
        for p in vault.rglob("*.md"):
            for target in re.findall(r"\\[\\^\\]+\\]", p.read_text(encoding="utf-8")):
                target_path = vault / (target + ".md")
                if target_path.exists():
                    if str(target_path) not in backlinks:
                        backlinks[str(target_path)] = []
                    backlinks[str(target_path)].append(str(p))

        for p in vault.rglob("*.md"):
            content = p.read_text(encoding="utf-8")
            content = re.sub(r"\n## Backlinks.*", "", content, flags=re.DOTALL)
            if str(p) in backlinks:
                content += "\n## Backlinks\n\n"
                for backlink in backlinks[str(p)]:
                    backlink_path = Path(backlink)
                    content += f"- [[{backlink_path.stem}]]\n"
            p.write_text(content, encoding="utf-8")
        log.append("Added backlinks to all notes.")
    if not tasks or "index" in tasks:
        index_path = vault / "Index.md"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("# Index\n\n")
            for p in sorted(vault.rglob("*.md")):
                if p.name != "Index.md":
                    f.write(f"- [[{p.stem}]]\n")
        log.append("Created index file.")
    return "\n".join(log)
