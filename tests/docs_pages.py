"""Read the user pages' code without running it.

The drift tests of the user pages share these helpers: cut a page into
sections and fenced blocks, and check its code against the installed gfem
without solving anything (the static page check: blocks parse, gfem names
resolve, call keywords bind to the signatures).
"""

import ast
import importlib
import inspect
import re
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"

_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<lang>[^\n`]*)\n(?P<body>.*?)^(?P=indent)```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_INLINE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_ROLE = re.compile(r"\{py:[a-z]+\}`([^`]+)`")
_DOTTED = re.compile(r"(?<![\w./-])gfem(?:\.[A-Za-z_]\w*)+")
# A dotted gfem name that is a file, not a Python name (`gfem.json`).
_FILE_SUFFIXES = {"json", "npz", "msh", "py", "md", "toml", "txt", "yml", "html"}


def user_pages() -> list[Path]:
    """Every user page with code a reader may copy, found on disk.

    The recipes are globbed, so a page is covered from the day it is added.
    """
    top = ("get-started.md", "conventions.md", "capabilities.md", "verification.md")
    pages = [DOCS / name for name in top if (DOCS / name).is_file()]
    return pages + sorted((DOCS / "recipes").glob("*.md"))


def section(page: str, heading: str) -> str:
    """The text under ``## heading`` up to the next ``## ``."""
    pattern = rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, page, re.MULTILINE | re.DOTALL)
    assert match, f"no section {heading!r} on the page"
    return match[1]


def blocks(text: str, lang: str) -> list[str]:
    """The bodies of the fenced blocks of one language, dedented."""
    return [
        textwrap.dedent(m["body"])
        for m in _FENCE.finditer(text)
        if m["lang"].strip() == lang
    ]


def line_formula(frequencies) -> np.ndarray:
    """|S21| of a 20 mm, eps_r = 10 parallel-plate section between matched lines."""
    c, eps, length = 299792458.0, 10.0, 20e-3
    z1 = 376.73 / 6
    z2 = z1 / np.sqrt(eps)
    bl = 2 * np.pi * np.asarray(frequencies) * np.sqrt(eps) / c * length
    s21 = 2 / (2 * np.cos(bl) + 1j * np.sin(bl) * (z2 / z1 + z1 / z2))
    return np.abs(s21)


def toctree_entries(page: str, caption: str | None = None) -> list[str]:
    """The documents of a page's toctrees (of the one with ``caption``, if given)."""
    entries = []
    for body in blocks(page, "{toctree}"):
        options = dict(re.findall(r"^:(\w+):[ \t]*(.*)$", body, re.MULTILINE))
        if caption is not None and options.get("caption") != caption:
            continue
        entries += [
            line.strip()
            for line in body.splitlines()
            if line.strip() and not line.startswith(":")
        ]
    return entries


def doc_links(text: str) -> set[str]:
    """The targets of the ``{doc}`` roles in ``text``."""
    targets = set()
    for role in re.findall(r"\{doc\}`([^`]+)`", text):
        explicit = re.search(r"<([^>]+)>\s*$", role)
        targets.add((explicit[1] if explicit else role).strip())
    return targets


# -- Static page check --------------------------------------------------------


class _MissingError(Exception):
    pass


def resolve(dotted: str):
    """The object a dotted name starting with ``gfem`` refers to.

    Raises `_MissingError` naming the first part that is not there. A dataclass
    field or annotated attribute without a class-level value counts as there.
    """
    parts = dotted.split(".")
    obj = importlib.import_module(parts[0])
    for i, part in enumerate(parts[1:], start=1):
        try:
            obj = getattr(obj, part)
            continue
        except AttributeError:
            pass
        if inspect.ismodule(obj):
            try:
                obj = importlib.import_module(".".join(parts[: i + 1]))
                continue
            except ImportError:
                pass
        declared = inspect.isclass(obj) and any(
            part in getattr(base, "__annotations__", {}) for base in obj.__mro__
        )
        if declared and i == len(parts) - 1:
            return None  # a declared attribute: nothing further to inspect
        raise _MissingError(f"{dotted}: no {part!r} in {'.'.join(parts[:i])}")
    return obj


def _chain(node: ast.AST) -> list[str] | None:
    """``["a", "b", "c"]`` for the expression ``a.b.c``; None for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id, *reversed(parts)]
    return None


class _Code:
    """The gfem names and calls of one page's Python, read from its syntax trees."""

    def __init__(self, bare: frozenset[str] = frozenset()):
        # Local name -> dotted gfem name. `bare` names (inline code written
        # without the `gfem.` prefix) resolve in the package namespace.
        self.aliases = {"gfem": "gfem"} | {name: f"gfem.{name}" for name in bare}
        self.problems: list[str] = []

    def read_imports(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] != "gfem":
                        continue
                    self._check(alias.name)
                    if alias.asname:
                        self.aliases[alias.asname] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module or ""
                if module.split(".")[0] != "gfem":
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    dotted = f"{module}.{alias.name}"
                    self._check(dotted)
                    self.aliases[alias.asname or alias.name] = dotted

    def forget_rebound(self, tree: ast.AST) -> None:
        """A name the page assigns to is the page's own, whatever was imported."""
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names = [node.id]
            elif isinstance(node, ast.arg):
                names = [node.arg]
            elif isinstance(node, ast.FunctionDef | ast.ClassDef):
                names = [node.name]
            for name in names:
                if name != "gfem":
                    self.aliases.pop(name, None)

    def _dotted(self, node: ast.AST) -> str | None:
        chain = _chain(node)
        if chain is None or chain[0] not in self.aliases:
            return None
        return ".".join([self.aliases[chain[0]], *chain[1:]])

    def _check(self, dotted: str):
        try:
            return resolve(dotted)
        except _MissingError as missing:
            self.problems.append(str(missing))
            return None

    def check(self, tree: ast.AST, *, complete_calls: bool) -> None:
        """Resolve every gfem name of ``tree`` and bind every gfem call.

        With ``complete_calls`` a call must also give every required argument
        (a code block); without, only the arguments it shows are checked
        (inline code, which abbreviates).
        """
        inner = {
            id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and id(node) not in inner:
                dotted = self._dotted(node)
                if dotted is not None:
                    self._check(dotted)
            if isinstance(node, ast.Call):
                dotted = self._dotted(node.func)
                if dotted is None:
                    continue
                try:
                    callee = resolve(dotted)
                except _MissingError:
                    continue  # reported as a name above (or below, for a bare name)
                if callable(callee):
                    self._bind(dotted, callee, node, complete=complete_calls)

    def _bind(self, dotted: str, callee, call: ast.Call, *, complete: bool) -> None:
        try:
            signature = inspect.signature(callee)
        except (TypeError, ValueError):
            return  # no signature to read (a builtin)
        starred = any(isinstance(arg, ast.Starred) for arg in call.args)
        unpacked = any(keyword.arg is None for keyword in call.keywords)
        positional = [] if starred else [None] * len(call.args)
        keywords = {k.arg: None for k in call.keywords if k.arg is not None}
        try:
            # The arguments shown must fit; in a complete call, none is missing.
            signature.bind_partial(*positional, **keywords)
            if complete and not starred and not unpacked:
                signature.bind(*positional, **keywords)
        except TypeError as error:
            shown = ast.unparse(call)
            self.problems.append(f"{shown}: {error} (signature {dotted}{signature})")


def _public_names() -> frozenset[str]:
    import gfem

    return frozenset(gfem.__all__)


def _prose(page: str) -> str:
    """The page without its fenced blocks."""
    return _FENCE.sub("", page)


def static_problems(page: str) -> list[str]:
    """What a reader copying from ``page`` would trip on, found without solving.

    - a ``python`` block that does not parse;
    - a ``gfem.`` name (in a code block, in inline code or in a ``{py:...}``
      role) that the package does not have;
    - a call of a gfem callable with a keyword its signature does not take,
      too many positional arguments or, in a code block, a required argument
      left out.

    The blocks of a page are read in order as one program: a name imported
    from gfem in one block is known in the next.
    """
    problems = []
    code = _Code()
    for number, block in enumerate(blocks(page, "python"), start=1):
        try:
            tree = ast.parse(block)
        except SyntaxError as error:
            problems.append(f"python block {number} does not parse: {error}")
            continue
        code.read_imports(tree)
        code.forget_rebound(tree)
        code.check(tree, complete_calls=True)
    problems += code.problems

    prose = _prose(page)
    inline = _Code(bare=_public_names())
    for text in _INLINE.findall(prose):
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError:
            tree = None
        if tree is not None:
            # A bare name is only read as gfem's when it is called or dotted:
            # `solve` alone may be an English word, `solve(...)` is not.
            inline.check(tree, complete_calls=False)
        else:
            for dotted in _DOTTED.findall(text):
                inline._check(dotted)
    problems += inline.problems

    roles = _Code()
    for role in _ROLE.findall(prose):
        explicit = re.search(r"<([^>]+)>\s*$", role)
        target = (explicit[1] if explicit else role).strip().lstrip("~!")
        if target.split(".")[0] == "gfem":
            roles._check(target.removesuffix("()"))
    problems += roles.problems

    def is_file(problem: str) -> bool:
        name = problem.split(":", maxsplit=1)[0]
        return name.rsplit(".", 1)[-1] in _FILE_SUFFIXES

    return sorted({p for p in problems if not is_file(p)})
