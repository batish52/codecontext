from __future__ import annotations

import ast
import fnmatch
import re
from pathlib import Path
from typing import Any


# Extensions handled by the regex-based JS/TS parser.
_JS_TS_EXTS = ('.ts', '.tsx', '.js', '.jsx')

# Candidate extensions we try when resolving a JS/TS relative import.
# Order matters: .ts/.tsx first so a TS project resolves to TS files over
# any emitted .js siblings.
_JS_TS_RESOLVE_EXTS = ('.ts', '.tsx', '.js', '.jsx')

# Identifiers we never treat as function-call callees, because they're
# language keywords / control-flow constructs that happen to be followed by
# '('. ``new`` is handled specially: ``new Foo()`` is recorded as a call to
# ``Foo`` (construction), but we skip ``new`` itself as a callee name.
_JS_TS_CALL_KEYWORDS = frozenset({
    'if', 'while', 'for', 'switch', 'catch', 'return', 'typeof', 'instanceof',
    'new', 'await', 'yield', 'function', 'throw', 'do', 'else', 'case',
    'delete', 'void', 'in', 'of', 'super', 'class', 'const', 'let', 'var',
    'import', 'export', 'default', 'from', 'as', 'type', 'async',
})


class ASTIndexer:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def _warn(self, message: str) -> None:
        self.warnings.append(message)

    def _is_js_ts_path(self, path: str) -> bool:
        return path.lower().endswith(_JS_TS_EXTS)

    def _path_to_module(self, path: str) -> str:
        # Python paths → dotted module names (as before).
        # JS/TS paths keep the extension and forward slashes so they round-trip
        # cleanly through _module_to_path. They are still valid dict keys and
        # can't collide with Python dotted names.
        norm = path.replace('\\', '/')
        if self._is_js_ts_path(norm):
            return norm
        return norm.removesuffix('.py').replace('/', '.')

    def _module_to_path(self, module: str) -> str:
        # For JS/TS, the "module name" already is a path.
        if self._is_js_ts_path(module) or '/' in module:
            return module
        return module.replace('.', '/') + '.py'

    def _ann_text(self, node: ast.AST | None) -> str | None:
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except Exception:
            return None

    def _call_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        if isinstance(node, ast.Call):
            return self._call_name(node.func)
        return None

    def _resolve_relative_module(self, current_module: str, module: str | None, level: int) -> str:
        parts = current_module.split('.') if current_module else []
        keep = max(0, len(parts) - level)
        base = parts[:keep]
        if module:
            base.extend(module.split('.'))
        return '.'.join(p for p in base if p)

    def index_file(self, path: Path) -> dict:
        # Dispatch to the right parser based on file extension.
        suffix = path.suffix.lower()
        if suffix in _JS_TS_EXTS:
            return self._index_js_ts_file(path)
        return self._index_python_file(path)

    def _index_python_file(self, path: Path) -> dict:
        try:
            text = path.read_text(encoding='utf-8')
        except Exception as exc:
            self._warn(f"read_failed:{path}:{exc}")
            return {"functions": [], "classes": [], "imports": [], "calls": []}

        try:
            tree = ast.parse(text, filename=str(path))
        except Exception as exc:
            self._warn(f"parse_failed:{path}:{exc}")
            return {"functions": [], "classes": [], "imports": [], "calls": []}

        functions: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []
        imports: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []

        class _Visitor(ast.NodeVisitor):
            def __init__(self, outer: ASTIndexer):
                self.outer = outer
                self.scope: list[str] = []
                self.class_scope: list[str] = []
                self.func_stack: list[dict[str, Any]] = []

            def _params(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
                args = []
                all_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
                for a in all_args:
                    args.append(a.arg)
                if node.args.vararg:
                    args.append('*' + node.args.vararg.arg)
                if node.args.kwarg:
                    args.append('**' + node.args.kwarg.arg)
                return args

            def _qual_name(self, name: str) -> str:
                return '.'.join(self.scope + [name]) if self.scope else name

            def visit_Import(self, node: ast.Import) -> Any:
                for alias in node.names:
                    imports.append({
                        "module": alias.name,
                        "names": [alias.asname or alias.name.split('.')[-1]],
                        "relative": False,
                        "level": 0,
                        "line": getattr(node, 'lineno', None),
                    })
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
                names = [a.asname or a.name for a in node.names]
                imports.append({
                    "module": node.module or "",
                    "names": names,
                    "relative": bool(getattr(node, 'level', 0) > 0),
                    "level": int(getattr(node, 'level', 0) or 0),
                    "line": getattr(node, 'lineno', None),
                })
                self.generic_visit(node)

            def visit_ClassDef(self, node: ast.ClassDef) -> Any:
                class_info = {
                    "name": node.name,
                    "qualified_name": self._qual_name(node.name),
                    "line": getattr(node, 'lineno', None),
                    "end_line": getattr(node, 'end_lineno', getattr(node, 'lineno', None)),
                    "bases": [self.outer._ann_text(b) or "" for b in node.bases],
                    "methods": [],
                }
                classes.append(class_info)
                self.scope.append(node.name)
                self.class_scope.append(node.name)
                self.generic_visit(node)
                self.class_scope.pop()
                self.scope.pop()

            def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                # Bug #8: previously this used `cls = class_scope[-1]`
                # (just the immediate parent class's short name) and
                # then `c["name"] == cls` to find the class to attach
                # the method to. When two classes in the same module
                # shared a short name — e.g. a top-level `class Config:`
                # and a nested `class Outer: class Config:` — the loop
                # broke on the first match, so methods of nested
                # `Outer.Config` got appended to the outer `Config` and
                # `Outer.Config` showed zero methods. Fix: build the
                # full enclosing-class qualified name (e.g. "Outer.Config")
                # and match against `qualified_name`, which uniquely
                # identifies each class record.
                enclosing_class_qn = ".".join(self.class_scope) if self.class_scope else None
                cls = self.class_scope[-1] if self.class_scope else None
                qn = self._qual_name(node.name)
                fn_info = {
                    "name": node.name,
                    "qualified_name": qn,
                    "class_name": cls,
                    "line": getattr(node, 'lineno', None),
                    "end_line": getattr(node, 'end_lineno', getattr(node, 'lineno', None)),
                    "params": self._params(node),
                    "returns": self.outer._ann_text(node.returns),
                    "decorators": [self.outer._ann_text(d) or "" for d in node.decorator_list],
                }
                functions.append(fn_info)
                if enclosing_class_qn:
                    for c in classes:
                        if c.get("qualified_name") == enclosing_class_qn:
                            c["methods"].append(node.name)
                            break

                self.scope.append(node.name)
                self.func_stack.append(fn_info)
                self.generic_visit(node)
                self.func_stack.pop()
                self.scope.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
                self._visit_function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
                self._visit_function(node)

            def visit_Call(self, node: ast.Call) -> Any:
                callee = self.outer._call_name(node.func)
                if self.func_stack and callee:
                    caller = self.func_stack[-1]
                    calls.append({
                        "caller_name": caller["name"],
                        "caller_qualified_name": caller["qualified_name"],
                        "caller_class": caller.get("class_name"),
                        "callee": callee,
                        "line": getattr(node, 'lineno', None),
                    })
                self.generic_visit(node)

        _Visitor(self).visit(tree)
        return {"functions": functions, "classes": classes, "imports": imports, "calls": calls}

    # ------------------------------------------------------------------
    # JS / TS regex-based parser
    # ------------------------------------------------------------------

    def _js_ts_sanitize(self, text: str) -> str:
        """Replace the contents of comments and string literals with spaces,
        preserving length and newlines. This lets subsequent regexes scan
        code without false matches inside strings or comments.

        Handles: // line comments, /* block comments */, '...' and "..."
        strings (with backslash escapes), and `...` template literals.
        Template literal ``${...}`` expressions keep their code intact (we
        only blank the string slices around them).
        """
        out = list(text)
        n = len(text)

        def blank(start: int, end: int) -> None:
            for k in range(start, min(end, n)):
                if out[k] != '\n':
                    out[k] = ' '

        i = 0
        while i < n:
            c = text[i]
            nxt = text[i + 1] if i + 1 < n else ''

            # Line comment
            if c == '/' and nxt == '/':
                j = text.find('\n', i + 2)
                end = n if j == -1 else j
                blank(i, end)
                i = end
                continue

            # Block comment
            if c == '/' and nxt == '*':
                j = text.find('*/', i + 2)
                end = n if j == -1 else j + 2
                blank(i, end)
                i = end
                continue

            # String literal (single or double quoted) — blank the interior,
            # keep the quote characters as-is so offsets stay aligned.
            if c == "'" or c == '"':
                quote = c
                j = i + 1
                while j < n:
                    cj = text[j]
                    if cj == '\\' and j + 1 < n:
                        j += 2
                        continue
                    if cj == '\n':  # unterminated string, bail out gracefully
                        break
                    if cj == quote:
                        j += 1
                        break
                    j += 1
                # Blank everything strictly between the opening and closing
                # quote positions.
                inner_end = max(i + 1, j - 1)
                blank(i + 1, inner_end)
                i = j
                continue

            # Template literal — blank the literal text, but recurse into any
            # ${...} expressions so identifiers inside them stay scannable.
            if c == '`':
                j = i + 1
                seg_start = j
                while j < n and text[j] != '`':
                    if text[j] == '\\' and j + 1 < n:
                        j += 2
                        continue
                    if text[j] == '$' and j + 1 < n and text[j + 1] == '{':
                        blank(seg_start, j)
                        depth = 1
                        k = j + 2
                        # Bug #7: this brace walker used to count only
                        # `{` and `}` characters, ignoring strings and
                        # comments inside the ${...} expression. So
                        # `${"}"}` (a legal template-expression
                        # containing a string with a `}`) closed at the
                        # `}` inside the string, leaving the rest of
                        # the template misinterpreted. The fix skips
                        # over string literals (' " `), line comments,
                        # block comments, and recursively-nested
                        # template literals while counting braces.
                        while k < n and depth > 0:
                            ck = text[k]
                            # Single-quoted or double-quoted string
                            if ck == "'" or ck == '"':
                                quote_ch = ck
                                k += 1
                                while k < n:
                                    if text[k] == '\\' and k + 1 < n:
                                        k += 2
                                        continue
                                    if text[k] == quote_ch:
                                        k += 1
                                        break
                                    if text[k] == '\n':
                                        # Unterminated; bail to outer
                                        # loop to keep us robust on
                                        # malformed source.
                                        break
                                    k += 1
                                continue
                            # Line comment
                            if ck == '/' and k + 1 < n and text[k + 1] == '/':
                                while k < n and text[k] != '\n':
                                    k += 1
                                continue
                            # Block comment
                            if ck == '/' and k + 1 < n and text[k + 1] == '*':
                                k += 2
                                while k + 1 < n and not (text[k] == '*' and text[k + 1] == '/'):
                                    k += 1
                                k = min(n, k + 2)
                                continue
                            # Nested template literal (legal: `${ `inner${1}` }`)
                            if ck == '`':
                                k += 1
                                while k < n and text[k] != '`':
                                    if text[k] == '\\' and k + 1 < n:
                                        k += 2
                                        continue
                                    if text[k] == '$' and k + 1 < n and text[k + 1] == '{':
                                        # Skip nested ${...} too,
                                        # tracking its own brace depth
                                        # independently.
                                        nested_depth = 1
                                        k += 2
                                        while k < n and nested_depth > 0:
                                            if text[k] == '{':
                                                nested_depth += 1
                                            elif text[k] == '}':
                                                nested_depth -= 1
                                            k += 1
                                        continue
                                    k += 1
                                if k < n:
                                    k += 1  # consume closing backtick
                                continue
                            if ck == '{':
                                depth += 1
                            elif ck == '}':
                                depth -= 1
                            k += 1
                        j = k
                        seg_start = j
                        continue
                    j += 1
                blank(seg_start, j)
                if j < n:
                    j += 1  # consume closing backtick
                i = j
                continue

            i += 1

        return ''.join(out)

    def _line_of(self, text: str, pos: int) -> int:
        return text.count('\n', 0, pos) + 1

    def _match_brace_end(self, text: str, open_pos: int) -> int:
        """Given the index of a '{', return the index just past the matching
        '}'. Returns len(text) if unbalanced. Input text is assumed already
        sanitized (strings/comments blanked)."""
        depth = 0
        n = len(text)
        i = open_pos
        while i < n:
            c = text[i]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return n

    def _match_paren_end(self, text: str, open_pos: int) -> int:
        """Given the index of a '(', return the index of the matching ')'.
        Returns -1 if unbalanced."""
        depth = 0
        n = len(text)
        i = open_pos
        while i < n:
            c = text[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return -1

    # Import / export patterns (operate on sanitized text).
    _RE_IMPORT_FROM = re.compile(
        r"""\bimport\s+(?P<type_kw>type\s+)?
            (?P<spec>
                (?:\*\s+as\s+\w+)              # * as ns
                | (?:\{[^}]*\})                # { named }
                | (?:\w+\s*,\s*\{[^}]*\})      # default, { named }
                | (?:\w+\s*,\s*\*\s+as\s+\w+)  # default, * as ns
                | (?:\w+)                      # default
            )
            \s+from\s*['"](?P<mod>[^'"]+)['"]""",
        re.VERBOSE,
    )
    _RE_IMPORT_SIDE_EFFECT = re.compile(
        r"""(^|[^.\w$])import\s*['"](?P<mod>[^'"]+)['"]"""
    )
    _RE_IMPORT_DYNAMIC = re.compile(
        r"""\bimport\s*\(\s*['"](?P<mod>[^'"]+)['"]\s*\)"""
    )
    _RE_EXPORT_FROM = re.compile(
        r"""\bexport\s+(?:type\s+)?(?:\*(?:\s+as\s+\w+)?|\{[^}]*\})\s+from\s*['"](?P<mod>[^'"]+)['"]"""
    )
    _RE_REQUIRE = re.compile(
        r"""(?<![\w$.])require\s*\(\s*['"](?P<mod>[^'"]+)['"]\s*\)"""
    )
    _RE_NAMED_SPEC = re.compile(
        r"""(?:^|,)\s*(?:type\s+)?(\w+)(?:\s+as\s+(\w+))?\s*"""
    )

    def _extract_js_ts_imports(self, sanitized: str, raw: str) -> list[dict[str, Any]]:
        """Extract import/export/require edges. Each entry mirrors the shape
        used by the Python path: {module, names, relative, level, line}.

        We match on sanitized text (strings/comments blanked) so regexes don't
        false-match inside string literals or comments, but we read module
        specifiers from the raw text — sanitizing blanks the interior of the
        quoted module path itself.
        """
        imports: list[dict[str, Any]] = []

        def _mod_from_span(m: 're.Match[str]') -> str:
            # Regex always names the module group 'mod' and its span covers
            # the literal module path (without surrounding quotes).
            s, e = m.span('mod')
            return raw[s:e]
        # Track byte spans we've already consumed so side-effect-import
        # regex doesn't double-count an `import X from 'mod'`.
        consumed_spans: list[tuple[int, int]] = []

        def _already_consumed(start: int, end: int) -> bool:
            for (a, b) in consumed_spans:
                if start < b and end > a:
                    return True
            return False

        def _names_from_spec(spec: str) -> list[str]:
            spec = spec.strip()
            names: list[str] = []

            # `* as ns`
            if spec.startswith('*'):
                m = re.match(r'\*\s+as\s+(\w+)', spec)
                if m:
                    names.append(m.group(1))
                return names

            # Bare `{ a, b as c }` — check BEFORE comma-split because a named
            # spec has commas inside the braces that must not be mistaken
            # for a default,rest separator.
            if spec.startswith('{'):
                inner = spec.strip('{}').strip()
                for m in self._RE_NAMED_SPEC.finditer(inner):
                    local = m.group(2) or m.group(1)
                    if local:
                        names.append(local)
                return names

            # `default, { named }` or `default, * as ns` — only split on the
            # FIRST comma that is at brace-depth 0.
            if ',' in spec:
                depth = 0
                split_at = -1
                for i, ch in enumerate(spec):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                    elif ch == ',' and depth == 0:
                        split_at = i
                        break
                if split_at != -1:
                    default_part = spec[:split_at].strip()
                    rest = spec[split_at + 1:].strip()
                    if re.fullmatch(r'\w+', default_part):
                        names.append(default_part)
                    if rest.startswith('{'):
                        inner = rest.strip('{}').strip()
                        for m in self._RE_NAMED_SPEC.finditer(inner):
                            local = m.group(2) or m.group(1)
                            if local:
                                names.append(local)
                    elif rest.startswith('*'):
                        m = re.match(r'\*\s+as\s+(\w+)', rest)
                        if m:
                            names.append(m.group(1))
                    return names

            # Single default identifier.
            if re.fullmatch(r'\w+', spec):
                names.append(spec)

            return names

        for m in self._RE_IMPORT_FROM.finditer(sanitized):
            mod = _mod_from_span(m)
            spec = m.group('spec')
            names = _names_from_spec(spec)
            is_type = bool(m.group('type_kw'))
            imports.append({
                "module": mod,
                "names": names,
                "relative": mod.startswith('.'),
                "level": 0,
                "line": self._line_of(sanitized, m.start()),
                "type_only": is_type,
                "kind": "import",
            })
            consumed_spans.append((m.start(), m.end()))

        for m in self._RE_IMPORT_DYNAMIC.finditer(sanitized):
            mod = _mod_from_span(m)
            imports.append({
                "module": mod,
                "names": [],
                "relative": mod.startswith('.'),
                "level": 0,
                "line": self._line_of(sanitized, m.start()),
                "type_only": False,
                "kind": "dynamic_import",
            })
            consumed_spans.append((m.start(), m.end()))

        for m in self._RE_EXPORT_FROM.finditer(sanitized):
            mod = _mod_from_span(m)
            imports.append({
                "module": mod,
                "names": [],
                "relative": mod.startswith('.'),
                "level": 0,
                "line": self._line_of(sanitized, m.start()),
                "type_only": False,
                "kind": "reexport",
            })
            consumed_spans.append((m.start(), m.end()))

        # Side-effect imports — must not overlap with any consumed span.
        for m in self._RE_IMPORT_SIDE_EFFECT.finditer(sanitized):
            if _already_consumed(m.start(), m.end()):
                continue
            mod = _mod_from_span(m)
            # Find the position of the `import` keyword (the prefix group
            # before the regex captures a non-word char).
            kw_pos = sanitized.find('import', m.start())
            if kw_pos == -1:
                kw_pos = m.start()
            imports.append({
                "module": mod,
                "names": [],
                "relative": mod.startswith('.'),
                "level": 0,
                "line": self._line_of(sanitized, kw_pos),
                "type_only": False,
                "kind": "import",
            })

        for m in self._RE_REQUIRE.finditer(sanitized):
            mod = _mod_from_span(m)
            imports.append({
                "module": mod,
                "names": [],
                "relative": mod.startswith('.'),
                "level": 0,
                "line": self._line_of(sanitized, m.start()),
                "type_only": False,
                "kind": "require",
            })

        return imports

    # Function / class declaration patterns.
    # The `name` group captures the declared identifier.
    _RE_FUNC_DECL = re.compile(
        r"""(?:\bexport\s+(?:default\s+)?)?(?:\basync\s+)?\bfunction\s*\*?\s*(?P<name>\w+)\s*(?:<[^>(]*>)?\s*\("""
    )
    _RE_ARROW_DECL = re.compile(
        r"""(?:\bexport\s+(?:default\s+)?)?\b(?:const|let|var)\s+(?P<name>\w+)\s*(?::\s*[^=;]+?)?\s*=\s*(?:async\s+)?(?:\((?P<plist>[^)]*)\)|(?P<single>\w+))\s*(?::\s*[^=;]+?)?\s*=>"""
    )
    _RE_FUNC_EXPR_DECL = re.compile(
        r"""(?:\bexport\s+(?:default\s+)?)?\b(?:const|let|var)\s+(?P<name>\w+)\s*(?::\s*[^=;]+?)?\s*=\s*(?:async\s+)?function\s*\*?\s*\w*\s*\("""
    )
    _RE_CLASS_DECL = re.compile(
        r"""(?:\bexport\s+(?:default\s+)?)?\bclass\s+(?P<name>\w+)(?:\s*<[^>{]*>)?(?:\s+extends\s+(?P<base>[\w.]+(?:\s*<[^>{]*>)?))?(?:\s+implements\s+[\w.,\s<>]+?)?\s*\{"""
    )
    # Method declaration inside a class body. Anchored at start of line.
    _RE_METHOD = re.compile(
        r"""^\s*(?:(?:public|private|protected|readonly|static|async|override|abstract)\s+)*(?:get\s+|set\s+)?(?:\*\s*)?(?P<name>\w+)\s*(?:<[^>(]*>)?\s*\((?P<plist>[^)]*)\)(?:\s*:\s*[^{;]+)?\s*\{""",
        re.MULTILINE,
    )

    _RE_CALL = re.compile(
        r"""(?<![\w$.])(?P<name>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\("""
    )

    @staticmethod
    def _split_params(raw: str) -> list[str]:
        """Parse a parameter list string into parameter names. Handles
        default values, destructuring, rest, and type annotations at a
        best-effort level — we only need the names."""
        if not raw or not raw.strip():
            return []
        out: list[str] = []
        depth = 0
        cur: list[str] = []
        for ch in raw:
            if ch in '<([{':
                depth += 1
                cur.append(ch)
            elif ch in '>)]}':
                depth -= 1
                cur.append(ch)
            elif ch == ',' and depth == 0:
                out.append(''.join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        tail = ''.join(cur).strip()
        if tail:
            out.append(tail)

        params: list[str] = []
        for p in out:
            if '=' in p:
                p = p.split('=', 1)[0].strip()
            # Strip TS type annotation (everything after first ':' at depth 0).
            # But keep destructuring braces/brackets where ':' means rename.
            if not (p.startswith('{') or p.startswith('[')):
                if ':' in p:
                    p = p.split(':', 1)[0].strip()
            rest = False
            if p.startswith('...'):
                rest = True
                p = p[3:].strip()
            if p.startswith('{') or p.startswith('['):
                params.append(('...' if rest else '') + p)
                continue
            if p.endswith('?'):
                p = p[:-1]
            if p:
                params.append(('*' + p) if rest else p)
        return params

    def _index_js_ts_file(self, path: Path) -> dict:
        try:
            raw = path.read_text(encoding='utf-8')
        except Exception as exc:
            self._warn(f"read_failed:{path}:{exc}")
            return {"functions": [], "classes": [], "imports": [], "calls": []}

        try:
            text = self._js_ts_sanitize(raw)
        except Exception as exc:
            self._warn(f"parse_failed:{path}:{exc}")
            return {"functions": [], "classes": [], "imports": [], "calls": []}

        functions: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        imports = self._extract_js_ts_imports(text, raw)

        # --- Collect function / class definitions with their body spans ---
        defs: list[dict[str, Any]] = []

        # `function foo(...) { }` — m.end() is right after the '(' that's
        # part of the regex.
        for m in self._RE_FUNC_DECL.finditer(text):
            name = m.group('name')
            paren_open = m.end() - 1  # position of '('
            paren_close = self._match_paren_end(text, paren_open)
            if paren_close == -1:
                continue
            params = self._split_params(text[paren_open + 1:paren_close])
            brace = text.find('{', paren_close)
            # A `function foo(): ReturnType { }` declaration may have a
            # return-type annotation between ')' and '{'. `text.find` will
            # still find the '{'. An ambient/overload signature ending in
            # `;` has no body — skip.
            if brace == -1:
                continue
            semi = text.find(';', paren_close)
            if 0 <= semi < brace:
                continue  # overload/ambient signature, no body
            end = self._match_brace_end(text, brace)
            defs.append({
                "kind": "function",
                "name": name,
                "class_name": None,
                "start": m.start(),
                "body_start": brace + 1,
                "body_end": end - 1,
                "line": self._line_of(text, m.start()),
                "end_line": self._line_of(text, end - 1),
                "params": params,
            })

        # `const foo = (...) => { ... }` and `const foo = x => expr`.
        for m in self._RE_ARROW_DECL.finditer(text):
            name = m.group('name')
            plist = m.group('plist')
            single = m.group('single')
            params = self._split_params(plist) if plist is not None else ([single] if single else [])
            j = m.end()
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] == '{':
                body_end = self._match_brace_end(text, j)
                defs.append({
                    "kind": "arrow",
                    "name": name,
                    "class_name": None,
                    "start": m.start(),
                    "body_start": j + 1,
                    "body_end": body_end - 1,
                    "line": self._line_of(text, m.start()),
                    "end_line": self._line_of(text, body_end - 1),
                    "params": params,
                })
            else:
                # Expression-bodied arrow: no block scope for nested calls.
                defs.append({
                    "kind": "arrow",
                    "name": name,
                    "class_name": None,
                    "start": m.start(),
                    "body_start": -1,
                    "body_end": -1,
                    "line": self._line_of(text, m.start()),
                    "end_line": self._line_of(text, m.start()),
                    "params": params,
                })

        # `const foo = function(...) { }`
        for m in self._RE_FUNC_EXPR_DECL.finditer(text):
            name = m.group('name')
            paren_open = m.end() - 1
            paren_close = self._match_paren_end(text, paren_open)
            if paren_close == -1:
                continue
            params = self._split_params(text[paren_open + 1:paren_close])
            brace = text.find('{', paren_close)
            if brace == -1:
                continue
            end = self._match_brace_end(text, brace)
            defs.append({
                "kind": "function_expr",
                "name": name,
                "class_name": None,
                "start": m.start(),
                "body_start": brace + 1,
                "body_end": end - 1,
                "line": self._line_of(text, m.start()),
                "end_line": self._line_of(text, end - 1),
                "params": params,
            })

        # Classes.
        class_spans: list[tuple[int, int, str]] = []
        for m in self._RE_CLASS_DECL.finditer(text):
            name = m.group('name')
            base = (m.group('base') or '').strip()
            brace = m.end() - 1  # position of '{'
            end = self._match_brace_end(text, brace)
            class_info = {
                "name": name,
                "qualified_name": name,
                "line": self._line_of(text, m.start()),
                "end_line": self._line_of(text, end - 1),
                "bases": [base] if base else [],
                "methods": [],
            }
            classes.append(class_info)
            class_spans.append((brace + 1, end - 1, name))

        # Methods inside class bodies.
        for body_start, body_end, class_name in class_spans:
            body = text[body_start:body_end]
            for m in self._RE_METHOD.finditer(body):
                name = m.group('name')
                if name in _JS_TS_CALL_KEYWORDS:
                    continue
                # The regex already matched up through '{', so m.end()-1 is
                # the '{' position within `body`.
                brace_rel = m.end() - 1
                end_rel = self._match_brace_end(body, brace_rel)
                abs_start = body_start + m.start()
                abs_body_start = body_start + brace_rel + 1
                abs_body_end = body_start + end_rel - 1
                params = self._split_params(m.group('plist') or '')
                defs.append({
                    "kind": "method",
                    "name": name,
                    "class_name": class_name,
                    "start": abs_start,
                    "body_start": abs_body_start,
                    "body_end": abs_body_end,
                    "line": self._line_of(text, abs_start),
                    "end_line": self._line_of(text, abs_body_end),
                    "params": params,
                })
                for c in classes:
                    if c["name"] == class_name and name not in c["methods"]:
                        c["methods"].append(name)
                        break

        # Emit function records in source order.
        defs.sort(key=lambda d: d['start'])
        for d in defs:
            cls = d.get('class_name')
            qn = f"{cls}.{d['name']}" if cls else d['name']
            functions.append({
                "name": d['name'],
                "qualified_name": qn,
                "class_name": cls,
                "line": d['line'],
                "end_line": d['end_line'],
                "params": d.get('params') or [],
                "returns": None,
                "decorators": [],
            })

        # --- Calls: attribute each call site to its innermost enclosing def ---
        bodied_defs = [d for d in defs if d['body_start'] >= 0]
        # Sort by body_start ascending so we can do a linear innermost-search.
        bodied_defs.sort(key=lambda d: d['body_start'])

        def _enclosing_def(pos: int) -> dict | None:
            enclosing: dict | None = None
            for d in bodied_defs:
                if d['body_start'] > pos:
                    break
                if d['body_start'] <= pos < d['body_end']:
                    if enclosing is None or d['body_start'] > enclosing['body_start']:
                        enclosing = d
            return enclosing

        for m in self._RE_CALL.finditer(text):
            callee = m.group('name')
            head = callee.split('.', 1)[0]
            if head in _JS_TS_CALL_KEYWORDS or callee in _JS_TS_CALL_KEYWORDS:
                continue
            start = m.start()
            # Skip if this match is really a definition site, e.g. the regex
            # would match `foo(` inside `function foo(` or `class Foo extends
            # Bar (` (which doesn't happen but be defensive) — any position
            # where some def starts exactly here.
            prefix = text[max(0, start - 20):start]
            if re.search(r'\bfunction\s*\*?\s*$', prefix):
                continue
            if re.search(r'\bclass\s+$', prefix):
                continue
            enc = _enclosing_def(start)
            if enc is None:
                continue
            # Don't record a function's own name as a call at its declaration
            # site. If the call position exactly matches any def's start, skip.
            if any(d['start'] == start for d in defs):
                continue
            caller_qn = (f"{enc['class_name']}.{enc['name']}"
                         if enc.get('class_name') else enc['name'])
            calls.append({
                "caller_name": enc['name'],
                "caller_qualified_name": caller_qn,
                "caller_class": enc.get('class_name'),
                "callee": callee,
                "line": self._line_of(text, start),
            })

        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "calls": calls,
        }

    # ------------------------------------------------------------------
    # Project-level indexing
    # ------------------------------------------------------------------

    def index_project(self, root: Path, exclude_patterns: list[str]) -> dict:
        root = root.resolve()
        out: dict[str, Any] = {
            "files": {},
            "warnings": self.warnings,
            "root": str(root),
        }
        patterns = ['*.py', '*.ts', '*.tsx', '*.js', '*.jsx']
        seen: set[str] = set()
        collected: list[Path] = []
        for pat in patterns:
            for path in root.rglob(pat):
                rel = path.relative_to(root).as_posix()
                if rel in seen:
                    continue
                seen.add(rel)
                collected.append(path)
        collected.sort(key=lambda p: p.relative_to(root).as_posix())
        for path in collected:
            rel = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, pat) for pat in (exclude_patterns or [])):
                continue
            out["files"][rel] = self.index_file(path)
        return out

    # ------------------------------------------------------------------
    # Graph building
    # ------------------------------------------------------------------

    def _resolve_js_ts_import(
        self, current_path: str, module: str, all_files: dict[str, Any]
    ) -> str | None:
        """Resolve a JS/TS import specifier against the indexed file set.
        Returns the matching file path or None for external packages or
        unresolved specifiers."""
        if not module:
            return None
        # External package (bare specifier): skip file resolution.
        if not (module.startswith('.') or module.startswith('/')):
            return None

        cur = current_path.replace('\\', '/')
        base_dir = cur.rsplit('/', 1)[0] if '/' in cur else ''
        parts_base = [p for p in base_dir.split('/') if p] if base_dir else []
        parts = module.split('/')
        stack = list(parts_base)
        for p in parts:
            if p == '' or p == '.':
                continue
            if p == '..':
                if stack:
                    stack.pop()
                continue
            stack.append(p)
        candidate = '/'.join(stack)

        if candidate in all_files:
            return candidate
        for ext in _JS_TS_RESOLVE_EXTS:
            cand = candidate + ext
            if cand in all_files:
                return cand
        for ext in _JS_TS_RESOLVE_EXTS:
            cand = f"{candidate}/index{ext}"
            if cand in all_files:
                return cand
        return None

    def _build_module_resolution(
        self, index: dict
    ) -> tuple[dict[str, str], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        module_to_path: dict[str, str] = {}
        module_aliases: dict[str, dict[str, str]] = {}
        imported_symbols: dict[str, dict[str, str]] = {}

        files = index.get("files") or {}

        for path, data in files.items():
            module_name = self._path_to_module(path)
            module_to_path[module_name] = path
            module_aliases[path] = {}
            imported_symbols[path] = {}

            is_jsts = self._is_js_ts_path(path)

            for imp in data.get("imports", []):
                mod_raw = str(imp.get("module") or "")
                level = int(imp.get("level") or 0)

                if is_jsts:
                    # JS/TS: resolve specifier to a real file when possible.
                    resolved = self._resolve_js_ts_import(path, mod_raw, files)
                    mod = resolved if resolved is not None else mod_raw
                else:
                    mod = mod_raw
                    if imp.get("relative"):
                        mod = self._resolve_relative_module(module_name, mod, level)

                names = imp.get("names") or []
                if names and mod:
                    for n in names:
                        if n == "*":
                            continue
                        imported_symbols[path][str(n)] = mod

                if mod:
                    if is_jsts:
                        last = mod.rsplit('/', 1)[-1]
                        alias = last.rsplit('.', 1)[0] if '.' in last else last
                    else:
                        alias = mod.split('.')[-1]
                    if alias:
                        module_aliases[path][alias] = mod
                    for n in names:
                        if n and n not in {"*"}:
                            module_aliases[path][str(n)] = mod

        return module_to_path, module_aliases, imported_symbols

    def build_call_graph(self, index: dict) -> dict:
        graph: dict[str, list[str]] = {}
        files = index.get("files") or {}

        def_map_by_path: dict[str, dict[str, str]] = {}
        global_name_map: dict[str, list[str]] = {}
        class_methods_by_path: dict[str, dict[str, str]] = {}

        for path, data in files.items():
            local_defs: dict[str, str] = {}
            methods: dict[str, str] = {}
            for fn in data.get("functions", []):
                name = str(fn.get("name") or "")
                qn = str(fn.get("qualified_name") or name)
                key = f"{path}:{qn}"
                if name:
                    local_defs[name] = key
                    global_name_map.setdefault(name, []).append(key)
                cls = fn.get("class_name")
                if cls and name:
                    methods[f"{cls}.{name}"] = key
            def_map_by_path[path] = local_defs
            class_methods_by_path[path] = methods

        module_to_path, module_aliases, imported_symbols = self._build_module_resolution(index)

        def resolve_callee(path: str, caller_class: str | None, callee: str) -> str:
            if not callee:
                return f"{path}:<unknown>"

            # `this.` in JS/TS plays the role of `self.` in Python.
            if callee.startswith('this.') and caller_class:
                m = callee.split('.', 1)[1]
                key = class_methods_by_path.get(path, {}).get(f"{caller_class}.{m}")
                if key:
                    return key
                return f"{path}:{caller_class}.{m}"

            if callee.startswith('self.') and caller_class:
                m = callee.split('.', 1)[1]
                key = class_methods_by_path.get(path, {}).get(f"{caller_class}.{m}")
                if key:
                    return key
                return f"{path}:{caller_class}.{m}"

            if '.' in callee:
                lhs, rhs = callee.split('.', 1)
                mod = module_aliases.get(path, {}).get(lhs)
                if mod:
                    mod_path = module_to_path.get(mod, self._module_to_path(mod))
                    return f"{mod_path}:{rhs}"

            local = def_map_by_path.get(path, {}).get(callee)
            if local:
                return local

            imported_mod = imported_symbols.get(path, {}).get(callee)
            if imported_mod:
                mod_path = module_to_path.get(imported_mod, self._module_to_path(imported_mod))
                return f"{mod_path}:{callee}"

            global_hits = global_name_map.get(callee, [])
            if len(global_hits) == 1:
                return global_hits[0]

            return f"{path}:{callee}"

        for path, data in files.items():
            for c in data.get("calls", []):
                caller_name = str(c.get("caller_qualified_name") or c.get("caller_name") or "")
                if not caller_name:
                    continue
                caller_key = f"{path}:{caller_name}"
                callee_key = resolve_callee(path, c.get("caller_class"), str(c.get("callee") or ""))
                graph.setdefault(caller_key, [])
                if callee_key not in graph[caller_key]:
                    graph[caller_key].append(callee_key)

        return graph

    def build_import_graph(self, index: dict) -> dict:
        graph: dict[str, list[str]] = {}
        files = index.get("files") or {}
        module_to_path, _, _ = self._build_module_resolution(index)

        for path, data in files.items():
            current_module = self._path_to_module(path)
            is_jsts = self._is_js_ts_path(path)
            deps: list[str] = []
            for imp in data.get("imports", []):
                mod_raw = str(imp.get("module") or "")
                level = int(imp.get("level") or 0)

                if is_jsts:
                    imported_path = self._resolve_js_ts_import(path, mod_raw, files)
                else:
                    mod = mod_raw
                    if imp.get("relative"):
                        mod = self._resolve_relative_module(current_module, mod, level)
                    if not mod:
                        continue
                    imported_path = module_to_path.get(mod)
                    if not imported_path:
                        candidate = self._module_to_path(mod)
                        if candidate in files:
                            imported_path = candidate

                if imported_path and imported_path != path and imported_path not in deps:
                    deps.append(imported_path)
            graph[path] = deps

        return graph

    def callers_of(self, call_graph: dict, target: str) -> list[str]:
        out: list[str] = []
        is_qualified = ':' in target
        target_name = target.split(':', 1)[1] if is_qualified else target
        for caller, callees in (call_graph or {}).items():
            for callee in (callees or []):
                callee_name = callee.split(':', 1)[1] if ':' in callee else callee
                if callee == target or (not is_qualified and callee_name.endswith(target_name)):
                    if caller not in out:
                        out.append(caller)
        return sorted(out)

    def callees_of(self, call_graph: dict, target: str) -> list[str]:
        if not call_graph:
            return []
        if ':' in target:
            return sorted(call_graph.get(target, []))
        out: list[str] = []
        for caller, callees in call_graph.items():
            caller_name = caller.split(':', 1)[1] if ':' in caller else caller
            if caller_name.endswith(target):
                for c in callees:
                    if c not in out:
                        out.append(c)
        return sorted(out)

    def dependents_of(self, import_graph: dict, target_path: str) -> list[str]:
        out = [p for p, deps in (import_graph or {}).items() if target_path in (deps or [])]
        return sorted(out)

    def dependencies_of(self, import_graph: dict, target_path: str) -> list[str]:
        return sorted((import_graph or {}).get(target_path, []))
