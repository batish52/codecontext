from __future__ import annotations

import ast
import fnmatch
from pathlib import Path
from typing import Any


class ASTIndexer:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def _warn(self, message: str) -> None:
        self.warnings.append(message)

    def _path_to_module(self, path: str) -> str:
        return path.replace('\\', '/').removesuffix('.py').replace('/', '.')

    def _module_to_path(self, module: str) -> str:
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
                if cls:
                    for c in classes:
                        if c["name"] == cls:
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

    def index_project(self, root: Path, exclude_patterns: list[str]) -> dict:
        root = root.resolve()
        out: dict[str, Any] = {
            "files": {},
            "warnings": self.warnings,
            "root": str(root),
        }
        for path in sorted(root.rglob('*.py')):
            rel = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, pat) for pat in (exclude_patterns or [])):
                continue
            out["files"][rel] = self.index_file(path)
        return out

    def _build_module_resolution(self, index: dict) -> tuple[dict[str, str], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        module_to_path: dict[str, str] = {}
        module_aliases: dict[str, dict[str, str]] = {}
        imported_symbols: dict[str, dict[str, str]] = {}

        for path, data in (index.get("files") or {}).items():
            module_name = self._path_to_module(path)
            module_to_path[module_name] = path
            module_aliases[path] = {}
            imported_symbols[path] = {}
            for imp in data.get("imports", []):
                mod = str(imp.get("module") or "")
                level = int(imp.get("level") or 0)
                if imp.get("relative"):
                    mod = self._resolve_relative_module(module_name, mod, level)

                names = imp.get("names") or []
                if names and mod:
                    for n in names:
                        if n == "*":
                            continue
                        imported_symbols[path][str(n)] = mod

                if mod:
                    alias = mod.split('.')[-1]
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
            deps: list[str] = []
            for imp in data.get("imports", []):
                mod = str(imp.get("module") or "")
                level = int(imp.get("level") or 0)
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
