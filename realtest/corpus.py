"""Get a real document set onto disk."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

K8S_REPO = "https://github.com/kubernetes/website.git"
K8S_SUBDIR = "content/en/docs"

_GLOSSARY = re.compile(r'\{\{<\s*glossary_tooltip[^>]*?text="([^"]*)"[^>]*>\}\}')
_SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.S)
_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def clean_hugo(text: str) -> str:
    """Kubernetes docs are Hugo pages: keep the title, drop front matter and
    template shortcodes, keep glossary link text."""
    title = ""
    m = _FRONT.match(text)
    if m:
        t = re.search(r"^title:\s*(.+)$", m.group(1), re.M)
        title = t.group(1).strip().strip("\"'") if t else ""
        text = text[m.end():]
    text = _GLOSSARY.sub(r"\1", text)
    text = _SHORTCODE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return (f"# {title}\n\n" if title else "") + text.strip() + "\n"


def fetch_k8s(dest: Path, log=print) -> Path:
    """Shallow, sparse clone of the Kubernetes docs (about 22 MB, 1,700 files),
    cleaned into dest/k8s-docs. Skipped if already there."""
    out = dest / "k8s-docs"
    if out.exists() and any(out.rglob("*.md")):
        return out
    src = dest / "_k8s-website"
    if not src.exists():
        log("Downloading the Kubernetes docs (shallow clone, about a minute)...")
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                        K8S_REPO, str(src)], check=True)
        subprocess.run(["git", "-C", str(src), "sparse-checkout", "set", K8S_SUBDIR], check=True)
    root = src / K8S_SUBDIR
    n = 0
    for f in root.rglob("*.md"):
        rel = f.relative_to(root)
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(clean_hugo(f.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
        n += 1
    shutil.rmtree(src, ignore_errors=True)
    log(f"Prepared {n} documents in {out}")
    return out
