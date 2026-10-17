"""The self-written skill library: a mutable system prompt, learned lessons, and helper modules.

Layout (all files under skills/ are written by the librarian agent, and versioned in git):
    skills/system_prompt.md   coder system prompt; the librarian may rewrite it
    skills/lessons.md         one learned rule per line ("- ...")
    skills/lib/*.py           reusable helper modules importable from scene code
"""

import ast
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Helper:
    module: str
    signatures: list[str]  # "def name(args) -> ret: first docstring line"


def _signatures(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    out = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_") and node.name != "demo":
            sig = f"def {node.name}({ast.unparse(node.args)})"
            if node.returns is not None:
                sig += f" -> {ast.unparse(node.returns)}"
            doc = (ast.get_docstring(node) or "").split("\n")[0]
            out.append(f"{sig}: {doc}" if doc else sig)
    return out


def helpers() -> list[Helper]:
    return [Helper(p.stem, _signatures(p)) for p in sorted(config.SKILL_LIB_DIR.glob("*.py"))]


def system_prompt() -> str:
    return (config.SKILLS_DIR / "system_prompt.md").read_text()


def lessons() -> list[str]:
    path = config.SKILLS_DIR / "lessons.md"
    return [l[2:].strip() for l in path.read_text().splitlines() if l.startswith("- ")]


def render_context() -> str:
    """The skill library as the coder sees it."""
    parts = ["## Helper modules you can import (from <module> import <name>)"]
    for h in helpers():
        parts.append(f"### {h.module}")
        parts.extend(f"    {s}" for s in h.signatures)
    parts.append("\n## Lessons learned from past critiques (follow them)")
    parts.extend(f"- {l}" for l in lessons())
    return "\n".join(parts)


def write_lessons(items: list[str]) -> None:
    (config.SKILLS_DIR / "lessons.md").write_text("".join(f"- {l}\n" for l in items))
