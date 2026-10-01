"""Every Python example in the docs uses the package's real public API.

The examples call real models, so they are read, not run. For each fenced
``python`` block in ``docs/``, plans excluded, the tests parse the block, check
that every name it imports from ``langchain_sync_monitors`` exists, and bind
each call of one of the package's classes or functions to its signature. An
unknown keyword, a positional argument to a keyword-only parameter and a
missing required argument all fail. A block that elides arguments with a bare
``...``, as in ``MonitorMiddleware(..., agent_name="worker")``, has only its
named keywords checked. The README's examples are checked the same way, and
every function or class a page's examples call must be defined or imported by
that point on the page, since a reader copies the blocks in order.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIRECTORY = REPOSITORY_ROOT / "docs"
WORKING_NOTE_DIRECTORIES = frozenset({"plans"})
PACKAGE_NAME = "langchain_sync_monitors"
PYTHON_FENCE_LANGUAGES = frozenset({"python", "py"})
OPENING_FENCE_PATTERN = re.compile(
    r"(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^`]*)",
)

type PackageBindings = dict[str, object]
"""The names a block binds to the package's modules, classes and functions."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CodeBlock:
    """One fenced Python block of a docs page, and the line its opening fence is on."""

    path: Path
    line_number: int
    source: str

    @property
    def label(self) -> str:
        """Name the block by its page and line, as a test id and in failure messages."""
        return f"{self.path.relative_to(REPOSITORY_ROOT)}:{self.line_number}"


def is_python_fence(info: str) -> bool:
    """Tell whether a fence's info string, such as `python title="agent.py"`, names Python."""
    words = info.split()
    return bool(words) and words[0].lower() in PYTHON_FENCE_LANGUAGES


def find_closing_fence(lines: list[str], *, start: int, fence: str) -> int:
    """Return the index of the line that closes `fence`, or the line count when none does.

    A closing fence uses the opening fence's character, at least as many times,
    and nothing else; an unclosed block runs to the end of the page.
    """
    closing = re.compile(rf"[ \t]*{re.escape(fence[0])}{{{len(fence)},}}[ \t]*")
    for index in range(start, len(lines)):
        if closing.fullmatch(lines[index]):
            return index
    return len(lines)


def read_python_blocks(text: str, *, path: Path) -> list[CodeBlock]:
    """Return every fenced Python block in a Markdown page, dedented, in page order.

    Blocks indented under an admonition or a list item are dedented, so they
    parse as the reader copies them.
    """
    lines = text.splitlines()
    blocks: list[CodeBlock] = []
    index = 0
    while index < len(lines):
        opening = OPENING_FENCE_PATTERN.fullmatch(lines[index])
        if opening is None:
            index += 1
            continue
        closing_index = find_closing_fence(lines, start=index + 1, fence=opening["fence"])
        if is_python_fence(opening["info"]):
            source = textwrap.dedent("\n".join(lines[index + 1 : closing_index]))
            blocks.append(CodeBlock(path=path, line_number=index + 1, source=source))
        index = closing_index + 1
    return blocks


def is_checked_page(path: Path) -> bool:
    """Tell whether a docs file is a page whose examples must match the API."""
    return path.suffix == ".md" and not WORKING_NOTE_DIRECTORIES.intersection(path.parts)


README_PATH = REPOSITORY_ROOT / "README.md"


def collect_checked_pages() -> list[Path]:
    """Return every docs page, plans excluded, and the README."""
    pages = sorted(path for path in DOCS_DIRECTORY.rglob("*.md") if is_checked_page(path))
    return [README_PATH, *pages]


def collect_docs_blocks() -> list[CodeBlock]:
    """Return the Python blocks of every checked page, in page order."""
    return [
        block
        for page in collect_checked_pages()
        for block in read_python_blocks(page.read_text(encoding="utf-8"), path=page)
    ]


def read_defined_names(tree: ast.Module) -> set[str]:
    """Return every name a block binds: imports, assignments, definitions and parameters."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def find_undefined_calls(tree: ast.Module, *, defined: set[str]) -> list[str]:
    """Return a message for every call of a bare name that nothing has defined or imported."""
    known = defined | read_defined_names(tree) | set(dir(builtins))
    return [
        f"line {node.lineno}: {node.func.id}(...) is called but never defined or imported"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id not in known
    ]


def find_page_undefined_calls(blocks: list[CodeBlock]) -> list[str]:
    """Check a page's blocks in order, each able to use what the blocks before it defined."""
    defined: set[str] = set()
    problems: list[str] = []
    for block in blocks:
        tree = ast.parse(block.source, filename=block.label)
        problems += [
            f"{block.label}: {problem}" for problem in find_undefined_calls(tree, defined=defined)
        ]
        defined |= read_defined_names(tree)
    return problems


def is_package_module(name: str | None) -> bool:
    """Tell whether an absolute module name is the package or one of its modules."""
    return name is not None and (name == PACKAGE_NAME or name.startswith(f"{PACKAGE_NAME}."))


def import_package_module(name: str) -> ModuleType | None:
    """Import one of the package's modules, or return `None` when it does not exist."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


def read_imported_object(module_name: str, *, name: str) -> object | None:
    """Return what `from <module_name> import <name>` gives, or `None` when that fails."""
    module = import_package_module(module_name)
    if module is None:
        return None
    if hasattr(module, name):
        return getattr(module, name)
    return import_package_module(f"{module_name}.{name}")


def is_package_import_from(node: ast.AST) -> bool:
    """Tell whether a node is an absolute `from ... import` of the package or its modules."""
    return isinstance(node, ast.ImportFrom) and node.level == 0 and is_package_module(node.module)


def find_import_problems(tree: ast.Module) -> list[str]:
    """Return a message for every module or name a block imports from the package in vain."""
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            problems += [
                f"line {node.lineno}: module {alias.name} does not exist"
                for alias in node.names
                if is_package_module(alias.name) and import_package_module(alias.name) is None
            ]
        if isinstance(node, ast.ImportFrom) and is_package_import_from(node):
            problems += find_missing_names(node)
    return problems


def find_missing_names(node: ast.ImportFrom) -> list[str]:
    """Return a message for every name a `from ... import` asks of the package in vain."""
    module_name = node.module or PACKAGE_NAME
    if import_package_module(module_name) is None:
        return [f"line {node.lineno}: module {module_name} does not exist"]
    return [
        f"line {node.lineno}: {module_name} has no name {alias.name!r}"
        for alias in node.names
        if alias.name != "*" and read_imported_object(module_name, name=alias.name) is None
    ]


def read_import_bindings(node: ast.Import) -> PackageBindings:
    """Map the names an `import` statement binds to the package modules it imports.

    `import a.b as c` binds `c` to `a.b`; `import a.b` binds `a` to `a`.
    """
    bindings: PackageBindings = {}
    for alias in node.names:
        imported_name = alias.name if alias.asname else alias.name.split(".")[0]
        module = import_package_module(imported_name) if is_package_module(alias.name) else None
        if module is not None:
            bindings[alias.asname or imported_name] = module
    return bindings


def read_import_from_bindings(node: ast.ImportFrom) -> PackageBindings:
    """Map the names a `from ... import` statement binds to the package objects it imports."""
    module_name = node.module or PACKAGE_NAME
    imported = {
        alias.asname or alias.name: read_imported_object(module_name, name=alias.name)
        for alias in node.names
    }
    return {name: target for name, target in imported.items() if target is not None}


def read_package_bindings(tree: ast.Module) -> PackageBindings:
    """Map each name a block imports from the package to the object it names."""
    bindings: PackageBindings = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bindings |= read_import_bindings(node)
        if isinstance(node, ast.ImportFrom) and is_package_import_from(node):
            bindings |= read_import_from_bindings(node)
    return bindings


def resolve_call_target(expression: ast.expr, *, bindings: PackageBindings) -> object | None:
    """Return the package object a call's function names, through module attributes only.

    `AutoMode(...)` and `monitors.AutoMode(...)` resolve; a method called on an
    instance does not, since its object is known only when the example runs.
    """
    if isinstance(expression, ast.Name):
        return bindings.get(expression.id)
    if isinstance(expression, ast.Attribute):
        owner = resolve_call_target(expression.value, bindings=bindings)
        if isinstance(owner, ModuleType):
            return getattr(owner, expression.attr, None)
    return None


def read_signature(target: object) -> inspect.Signature | None:
    """Return the signature of one of the package's classes or functions, else `None`.

    Exceptions, warnings and `TypedDict` records have no signature to check.
    """
    is_callable_api = inspect.isclass(target) or inspect.isfunction(target)
    if not is_callable_api or not is_package_module(getattr(target, "__module__", None)):
        return None
    try:
        return inspect.signature(target)
    except ValueError:
        return None


def is_elided_argument(argument: ast.expr) -> bool:
    """Tell whether a positional argument is unpacked, `*args`, or elided, a bare `...`."""
    is_ellipsis = isinstance(argument, ast.Constant) and argument.value is Ellipsis
    return is_ellipsis or isinstance(argument, ast.Starred)


def find_binding_problem(call: ast.Call, *, signature: inspect.Signature) -> str | None:
    """Bind a call's arguments to a signature, and return why they do not fit, if they do not.

    When the call unpacks or elides arguments, only its named keywords are checked.
    """
    named = {keyword.arg: None for keyword in call.keywords if keyword.arg is not None}
    is_partial = len(named) < len(call.keywords) or any(map(is_elided_argument, call.args))
    try:
        if is_partial:
            signature.bind_partial(**named)
        else:
            signature.bind(*[None] * len(call.args), **named)
    except TypeError as error:
        return str(error)
    return None


def find_call_problems(tree: ast.Module) -> list[str]:
    """Return a message for every call of the package's API that does not fit its signature."""
    bindings = read_package_bindings(tree)
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = resolve_call_target(node.func, bindings=bindings)
        signature = read_signature(target)
        problem = None if signature is None else find_binding_problem(node, signature=signature)
        if problem is not None:
            problems.append(f"line {node.lineno}: {ast.unparse(node.func)}(...): {problem}")
    return problems


def check_source(source: str, *, check: Callable[[ast.Module], list[str]]) -> list[str]:
    """Parse a snippet and run one check on it, for the tests of the checks themselves."""
    return check(ast.parse(source))


DOCS_BLOCKS = collect_docs_blocks()


CHECKED_PAGES = collect_checked_pages()


@pytest.mark.parametrize(
    "page",
    CHECKED_PAGES,
    ids=[str(page.relative_to(REPOSITORY_ROOT)) for page in CHECKED_PAGES],
)
def test_every_name_a_page_calls_is_defined_or_imported_first(page: Path) -> None:
    # Arrange
    blocks = read_python_blocks(page.read_text(encoding="utf-8"), path=page)

    # Act
    problems = find_page_undefined_calls(blocks)

    # Assert
    assert problems == []


def test_a_call_to_a_name_never_imported_is_reported() -> None:
    # Arrange: the second block calls AutoMode and create_agent, and imports neither.
    page = REPOSITORY_ROOT / "docs" / "example.md"
    blocks = [
        CodeBlock(
            path=page, line_number=1, source="from langchain_sync_monitors import LLMMonitor"
        ),
        CodeBlock(
            path=page,
            line_number=5,
            source="judge = LLMMonitor(model='m')\nagent = create_agent(protocol=AutoMode())",
        ),
    ]

    # Act
    problems = find_page_undefined_calls(blocks)

    # Assert
    assert len(problems) == 2
    assert any("AutoMode(...)" in problem for problem in problems)
    assert any("create_agent(...)" in problem for problem in problems)


def test_the_docs_hold_python_examples_to_check() -> None:
    # Act
    pages = {block.path.name for block in DOCS_BLOCKS}

    # Assert
    assert "index.md" in pages


@pytest.mark.parametrize("block", DOCS_BLOCKS, ids=[block.label for block in DOCS_BLOCKS])
def test_every_name_a_docs_example_imports_from_the_package_exists(block: CodeBlock) -> None:
    # Arrange
    tree = ast.parse(block.source, filename=block.label)

    # Act
    problems = find_import_problems(tree)

    # Assert
    assert problems == [], f"{block.label}: {problems}"


@pytest.mark.parametrize("block", DOCS_BLOCKS, ids=[block.label for block in DOCS_BLOCKS])
def test_every_package_call_in_a_docs_example_fits_its_signature(block: CodeBlock) -> None:
    # Arrange
    tree = ast.parse(block.source, filename=block.label)

    # Act
    problems = find_call_problems(tree)

    # Assert
    assert problems == [], f"{block.label}: {problems}"


def test_a_correct_example_has_no_problems() -> None:
    # Arrange
    source = """
import langchain_sync_monitors as sync_monitors
from langchain_sync_monitors import AutoMode, HaltRun as Halt, MonitorMiddleware

protocol = AutoMode(block_threshold=0.7, when_limit_reached=Halt(message="Stopped."))
middleware = MonitorMiddleware(monitor=judge, protocol=protocol)
fallback = sync_monitors.DeferToTrustedModel(trusted_model="openrouter:xiaomi/mimo-v2.6-flash")
"""

    # Act
    problems = [
        *check_source(source, check=find_import_problems),
        *check_source(source, check=find_call_problems),
    ]

    # Assert
    assert problems == []


def test_a_name_the_package_does_not_export_is_reported() -> None:
    # Act
    problems = check_source(
        "from langchain_sync_monitors import AutoMode, LLMJudge",
        check=find_import_problems,
    )

    # Assert
    assert problems == ["line 1: langchain_sync_monitors has no name 'LLMJudge'"]


def test_a_module_the_package_does_not_have_is_reported() -> None:
    # Act
    problems = check_source(
        "import langchain_sync_monitors.judges\nfrom langchain_sync_monitors.guards import Guard",
        check=find_import_problems,
    )

    # Assert
    assert problems == [
        "line 1: module langchain_sync_monitors.judges does not exist",
        "line 2: module langchain_sync_monitors.guards does not exist",
    ]


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ("AutoMode(block_limit=3)", "unexpected keyword argument 'block_limit'"),
        ("AutoMode(0.6)", "too many positional arguments"),
        (
            "DeferToResample(defer_threshold=0.6)",
            "missing a required keyword-only argument: 'fallback'",
        ),
        ("sync_monitors.HaltRun(text='Stopped.')", "unexpected keyword argument 'text'"),
    ],
)
def test_a_call_that_does_not_fit_its_signature_is_reported(call: str, expected: str) -> None:
    # Arrange
    source = (
        "import langchain_sync_monitors as sync_monitors\n"
        f"from langchain_sync_monitors import AutoMode, DeferToResample\n{call}"
    )

    # Act
    problems = check_source(source, check=find_call_problems)

    # Assert
    assert len(problems) == 1
    assert expected in problems[0]


@pytest.mark.parametrize(
    ("arguments", "expected_problems"),
    [
        ("..., agent_name='worker'", 0),
        ("..., agent='worker'", 1),
        ("**options", 0),
        ("*arguments, label='guard'", 0),
    ],
)
def test_elided_or_unpacked_arguments_leave_only_the_named_keywords_to_check(
    arguments: str,
    expected_problems: int,
) -> None:
    # Arrange
    source = (
        f"from langchain_sync_monitors import MonitorMiddleware\nMonitorMiddleware({arguments})"
    )

    # Act
    problems = check_source(source, check=find_call_problems)

    # Assert
    assert len(problems) == expected_problems


def test_calls_the_checks_cannot_resolve_are_left_alone() -> None:
    # Arrange: an instance method, a user's own class, an exception and a TypedDict record
    source = """
from langchain_sync_monitors import ConfigurationError, StepRecord

monitor.evaluate_sync(anything=1)
MyMonitor(whatever=2)
ConfigurationError("bad option")
StepRecord(agent="main")
"""

    # Act
    problems = check_source(source, check=find_call_problems)

    # Assert
    assert problems == []


def test_python_blocks_are_read_from_every_fence_style_and_dedented() -> None:
    # Arrange
    page = DOCS_DIRECTORY / "example.md"
    text = """# Example

```python
first = 1
```

!!! note
    ```py title="second.py"
    second = 2
    ```

````python
text = '''
```
'''
````

```console
$ not python
```

~~~python
unclosed = 3
"""

    # Act
    blocks = read_python_blocks(text, path=page)

    # Assert
    assert [(block.line_number, block.source) for block in blocks] == [
        (3, "first = 1"),
        (8, "second = 2"),
        (12, "text = '''\n```\n'''"),
        (22, "unclosed = 3"),
    ]


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    [
        ("how-to/use-auto-mode.md", True),
        ("plans/initial-implementation/README.md", False),
        ("references.bib", False),
    ],
)
def test_plans_and_non_pages_are_not_checked(relative_path: str, expected: bool) -> None:
    # Act
    checked = is_checked_page(DOCS_DIRECTORY / relative_path)

    # Assert
    assert checked is expected
