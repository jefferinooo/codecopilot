"""AST-aware chunker for Python source files."""
from __future__ import annotations

from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython

from .base import Chunk

PY_LANGUAGE = Language(tspython.language())


class PythonChunker:
    """Emit one chunk per top-level function, class, and method.

    Class chunks include the signature, decorators, and the class-level
    docstring. Method chunks include any decorators and any leading
    comment lines that immediately precede the method, so documentation
    stays attached to the code it describes. Module-level code not covered
    by any function or class is emitted as 'window' chunks at the end.
    """

    def __init__(self) -> None:
        self.parser = Parser(PY_LANGUAGE)

    def chunk(self, source: bytes) -> list[Chunk]:
        tree = self.parser.parse(source)
        imports = self._extract_imports(tree.root_node, source)

        chunks: list[Chunk] = []
        covered: list[tuple[int, int]] = []

        def visit(node: Node, class_name: str | None = None) -> None:
            # `decorated_definition` wraps a def/class with its decorators.
            inner = node
            if node.type == "decorated_definition":
                for child in node.children:
                    if child.type in ("function_definition", "class_definition"):
                        inner = child
                        break

            if inner.type == "function_definition":
                name = self._child_identifier(inner, source) or "<anonymous>"
                qualified = f"{class_name}.{name}" if class_name else name
                # Extend backwards over leading comment lines and blank lines
                start_byte = self._extend_back_over_comments(source, node.start_byte)
                chunks.append(Chunk(
                    content=source[start_byte:node.end_byte].decode("utf-8", "replace"),
                    start_line=source[:start_byte].count(b"\n") + 1,
                    end_line=node.end_point[0] + 1,
                    symbol_name=qualified,
                    symbol_kind="method" if class_name else "function",
                    imports=list(imports),
                ))
                covered.append((start_byte, node.end_byte))
                return

            if inner.type == "class_definition":
                cname = self._child_identifier(inner, source) or "<anonymous>"
                body = inner.child_by_field_name("body")
                start_byte = self._extend_back_over_comments(source, node.start_byte)

                # Include the class docstring (first statement in body, if it's a string)
                header_end = body.start_byte if body else inner.end_byte
                if body is not None:
                    docstring_end = self._docstring_end(body, source)
                    if docstring_end is not None:
                        header_end = docstring_end

                chunks.append(Chunk(
                    content=source[start_byte:header_end].decode("utf-8", "replace"),
                    start_line=source[:start_byte].count(b"\n") + 1,
                    end_line=source[:header_end].count(b"\n") + 1,
                    symbol_name=cname,
                    symbol_kind="class",
                    imports=list(imports),
                ))
                covered.append((start_byte, header_end))
                if body is not None:
                    for child in body.children:
                        visit(child, class_name=cname)
                return

            for child in node.children:
                visit(child, class_name=class_name)

        visit(tree.root_node)
        chunks.extend(self._fill_gaps(source, covered, imports))
        chunks.sort(key=lambda c: c.start_line)
        return chunks

    # ---------- helpers ----------

    @staticmethod
    def _child_identifier(node: Node, source: bytes) -> str | None:
        for child in node.children:
            if child.type == "identifier":
                return source[child.start_byte:child.end_byte].decode("utf-8", "replace")
        return None

    @staticmethod
    def _extract_imports(root: Node, source: bytes) -> list[str]:
        imports: list[str] = []
        for child in root.children:
            if child.type in ("import_statement", "import_from_statement"):
                imports.append(
                    source[child.start_byte:child.end_byte].decode("utf-8", "replace").strip()
                )
        return imports

    @staticmethod
    def _docstring_end(body: Node, source: bytes) -> int | None:
        """Return end_byte of body's leading docstring, or None if absent."""
        for child in body.children:
            if child.type in ("comment",):
                continue
            # First non-comment statement: is it a bare string expression?
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        return child.end_byte
            return None
        return None

    @staticmethod
    def _extend_back_over_comments(source: bytes, start: int) -> int:
        """Walk backwards from `start` over any contiguous comment lines.

        We extend as long as each preceding non-blank line begins with `#`.
        Blank lines are skipped (so `# comment\\n\\ndef foo` still attaches).
        Stops as soon as we hit a non-blank, non-comment line.
        """
        # Find the start of the line containing `start`
        line_start = source.rfind(b"\n", 0, start) + 1
        cursor = line_start
        while cursor > 0:
            prev_line_end = cursor - 1               # the '\n' before us
            prev_line_start = source.rfind(b"\n", 0, prev_line_end) + 1
            line = source[prev_line_start:prev_line_end].lstrip()
            if not line:                             # blank line -> keep walking
                cursor = prev_line_start
                continue
            if line.startswith(b"#"):                # comment -> include it
                cursor = prev_line_start
                continue
            break                                    # something else -> stop
        return cursor

    @staticmethod
    def _fill_gaps(
        source: bytes,
        covered: list[tuple[int, int]],
        imports: list[str],
    ) -> list[Chunk]:
        """Emit 'window' chunks for byte ranges not covered by AST chunks."""
        covered_sorted = sorted(covered)

        gaps: list[tuple[int, int]] = []
        cursor = 0
        for start, end in covered_sorted:
            if start > cursor:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < len(source):
            gaps.append((cursor, len(source)))

        chunks: list[Chunk] = []
        for start, end in gaps:
            text = source[start:end].decode("utf-8", "replace")
            if not text.strip():
                continue

            stripped = text.strip("\n")
            leading_newlines = len(text) - len(text.lstrip("\n"))
            real_start = start + leading_newlines

            start_line = source[:real_start].count(b"\n") + 1
            end_line = start_line + stripped.count("\n")

            chunks.append(Chunk(
                content=stripped,
                start_line=start_line,
                end_line=end_line,
                symbol_name=None,
                symbol_kind="window",
                imports=list(imports),
            ))
        return chunks
