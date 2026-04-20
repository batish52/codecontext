"""Tests for the ASTIndexer covering TypeScript/JavaScript parsing and
proving Python parsing behavior is unchanged.

The Python parity test is the most important one here: TS/JS support was
added by *extending* the indexer, and it must not perturb any Python result.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from codecontext.ast_graph import ASTIndexer


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content).lstrip('\n'), encoding='utf-8')


def _make_ts_project(root: Path) -> None:
    """A realistic React/TS + Node/Express mini-project exercising imports,
    classes, methods, arrow functions, CommonJS require, re-exports, and
    cross-file calls."""
    _write(root, 'src/utils/math.ts', """
        export function add(a: number, b: number): number {
            return a + b;
        }

        export function subtract(a: number, b: number): number {
            return a - b;
        }

        export const multiply = (a: number, b: number): number => {
            return a * b;
        };

        export default function divide(a: number, b: number): number {
            return a / b;
        }
    """)

    _write(root, 'src/utils/format.ts', """
        import { add } from './math';

        export function formatSum(x: number, y: number): string {
            const result = add(x, y);
            return `Sum: ${result}`;
        }

        export const formatProduct = (x: number, y: number): string => {
            return `Product: ${x * y}`;
        };
    """)

    _write(root, 'src/components/Calculator.tsx', """
        import React from 'react';
        import { add, multiply } from '../utils/math';
        import type { ReactNode } from 'react';
        import divide from '../utils/math';

        interface Props {
            x: number;
            y: number;
        }

        export class Calculator extends React.Component<Props> {
            private cache: Map<string, number> = new Map();

            constructor(props: Props) {
                super(props);
                this.cache = new Map();
            }

            compute(): number {
                const sum = add(this.props.x, this.props.y);
                const product = multiply(this.props.x, this.props.y);
                return this.combine(sum, product);
            }

            private combine(a: number, b: number): number {
                return divide(a, b);
            }

            render(): ReactNode {
                return <div>{this.compute()}</div>;
            }
        }
    """)

    _write(root, 'src/index.ts', """
        export { formatSum } from './utils/format';
        export * from './utils/math';
        import { Calculator } from './components/Calculator';

        const legacy = require('./utils/format');

        function main() {
            const c = new Calculator({ x: 1, y: 2 });
            console.log(legacy.formatProduct(3, 4));
        }

        main();
    """)

    _write(root, 'server.js', """
        const express = require('express');
        const { add } = require('./src/utils/math');

        const app = express();

        app.get('/add', function handler(req, res) {
            const result = add(Number(req.query.a), Number(req.query.b));
            res.json({ result });
        });

        app.listen(3000);
    """)


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------


def test_ts_parser_extracts_function_definitions(tmp_path: Path) -> None:
    _write(tmp_path, 'math.ts', """
        export function add(a: number, b: number): number {
            return a + b;
        }

        export const multiply = (a: number, b: number): number => {
            return a * b;
        };
    """)

    result = ASTIndexer().index_file(tmp_path / 'math.ts')

    names = {f['name'] for f in result['functions']}
    assert names == {'add', 'multiply'}

    add = next(f for f in result['functions'] if f['name'] == 'add')
    assert add['params'] == ['a', 'b']
    assert add['line'] == 1
    assert add['class_name'] is None


def test_ts_parser_extracts_default_export_function(tmp_path: Path) -> None:
    _write(tmp_path, 'm.ts', """
        export default function divide(a: number, b: number): number {
            return a / b;
        }
    """)

    result = ASTIndexer().index_file(tmp_path / 'm.ts')

    assert len(result['functions']) == 1
    assert result['functions'][0]['name'] == 'divide'


def test_ts_parser_extracts_class_with_methods(tmp_path: Path) -> None:
    _write(tmp_path, 'c.ts', """
        export class Calculator {
            constructor(public x: number) {}

            compute(): number {
                return this.x * 2;
            }

            private helper(v: number): number {
                return v + 1;
            }
        }
    """)

    result = ASTIndexer().index_file(tmp_path / 'c.ts')

    assert len(result['classes']) == 1
    cls = result['classes'][0]
    assert cls['name'] == 'Calculator'
    assert set(cls['methods']) == {'constructor', 'compute', 'helper'}

    methods = {(f['class_name'], f['name']) for f in result['functions']}
    assert methods == {('Calculator', 'constructor'),
                       ('Calculator', 'compute'),
                       ('Calculator', 'helper')}


def test_ts_parser_extracts_class_with_extends(tmp_path: Path) -> None:
    _write(tmp_path, 'c.tsx', """
        import React from 'react';
        export class Widget extends React.Component<Props> {
            render() { return null; }
        }
    """)

    result = ASTIndexer().index_file(tmp_path / 'c.tsx')

    cls = result['classes'][0]
    assert cls['name'] == 'Widget'
    assert cls['bases'] and 'React.Component' in cls['bases'][0]


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


def test_ts_parser_extracts_named_imports(tmp_path: Path) -> None:
    _write(tmp_path, 'a.ts', """
        import { add, multiply as mul } from './math';
    """)

    result = ASTIndexer().index_file(tmp_path / 'a.ts')

    assert len(result['imports']) == 1
    imp = result['imports'][0]
    assert imp['module'] == './math'
    assert set(imp['names']) == {'add', 'mul'}
    assert imp['relative'] is True


def test_ts_parser_extracts_default_and_type_imports(tmp_path: Path) -> None:
    _write(tmp_path, 'a.tsx', """
        import React from 'react';
        import type { ReactNode } from 'react';
        import divide from './math';
    """)

    result = ASTIndexer().index_file(tmp_path / 'a.tsx')

    by_line = {imp['line']: imp for imp in result['imports']}

    assert by_line[1]['module'] == 'react'
    assert by_line[1]['names'] == ['React']
    assert by_line[1]['type_only'] is False
    assert by_line[1]['relative'] is False

    assert by_line[2]['module'] == 'react'
    assert by_line[2]['names'] == ['ReactNode']
    assert by_line[2]['type_only'] is True

    assert by_line[3]['module'] == './math'
    assert by_line[3]['names'] == ['divide']
    assert by_line[3]['relative'] is True


def test_ts_parser_extracts_require_calls(tmp_path: Path) -> None:
    _write(tmp_path, 'a.js', """
        const express = require('express');
        const { add } = require('./math');
    """)

    result = ASTIndexer().index_file(tmp_path / 'a.js')

    modules = sorted(imp['module'] for imp in result['imports'])
    kinds = {imp['kind'] for imp in result['imports']}
    assert modules == ['./math', 'express']
    assert kinds == {'require'}


def test_ts_parser_extracts_reexports(tmp_path: Path) -> None:
    _write(tmp_path, 'a.ts', """
        export { foo } from './a';
        export * from './b';
    """)

    result = ASTIndexer().index_file(tmp_path / 'a.ts')

    kinds = [imp['kind'] for imp in result['imports']]
    modules = [imp['module'] for imp in result['imports']]
    assert kinds == ['reexport', 'reexport']
    assert sorted(modules) == ['./a', './b']


def test_ts_parser_ignores_imports_inside_strings_and_comments(tmp_path: Path) -> None:
    _write(tmp_path, 'a.ts', """
        // import { nope } from 'nope';
        const fake = "import { fake } from 'nowhere'";
        /* import { alsoFake } from 'alsofake'; */
        const r = "require('also-fake')";

        import { real } from './actual';
    """)

    result = ASTIndexer().index_file(tmp_path / 'a.ts')

    assert len(result['imports']) == 1
    assert result['imports'][0]['module'] == './actual'
    assert result['imports'][0]['names'] == ['real']


# ---------------------------------------------------------------------------
# Call attribution
# ---------------------------------------------------------------------------


def test_ts_parser_attributes_calls_to_enclosing_function(tmp_path: Path) -> None:
    _write(tmp_path, 'a.ts', """
        function outer() {
            helper();
            other();
        }

        function inner() {
            different();
        }
    """)

    result = ASTIndexer().index_file(tmp_path / 'a.ts')

    by_caller: dict[str, set[str]] = {}
    for c in result['calls']:
        by_caller.setdefault(c['caller_qualified_name'], set()).add(c['callee'])

    assert by_caller['outer'] == {'helper', 'other'}
    assert by_caller['inner'] == {'different'}


def test_ts_parser_attributes_method_calls_with_this(tmp_path: Path) -> None:
    _write(tmp_path, 'a.ts', """
        class Foo {
            a() { this.b(); }
            b() { this.c(); }
            c() { return 1; }
        }
    """)

    result = ASTIndexer().index_file(tmp_path / 'a.ts')

    edges = {(c['caller_qualified_name'], c['callee']) for c in result['calls']}
    assert ('Foo.a', 'this.b') in edges
    assert ('Foo.b', 'this.c') in edges


def test_ts_parser_skips_control_flow_keywords_as_callees(tmp_path: Path) -> None:
    _write(tmp_path, 'a.ts', """
        function f(xs: number[]) {
            if (xs.length > 0) {
                for (const x of xs) {
                    while (x > 0) {
                        real(x);
                    }
                }
            }
        }
    """)

    result = ASTIndexer().index_file(tmp_path / 'a.ts')

    callees = {c['callee'] for c in result['calls']}
    # Control keywords must not appear as callees.
    assert 'if' not in callees
    assert 'for' not in callees
    assert 'while' not in callees
    # The real call is recorded.
    assert 'real' in callees


# ---------------------------------------------------------------------------
# Cross-file graph building
# ---------------------------------------------------------------------------


def test_import_graph_resolves_relative_ts_paths(tmp_path: Path) -> None:
    _make_ts_project(tmp_path)

    idx = ASTIndexer()
    index = idx.index_project(tmp_path, [])
    graph = idx.build_import_graph(index)

    # format.ts imports ./math → math.ts
    assert 'src/utils/math.ts' in graph['src/utils/format.ts']

    # Calculator.tsx imports ../utils/math — deduplicated to single edge
    assert graph['src/components/Calculator.tsx'] == ['src/utils/math.ts']

    # index.ts pulls in all three via re-exports, named import, and require.
    assert set(graph['src/index.ts']) == {
        'src/components/Calculator.tsx',
        'src/utils/format.ts',
        'src/utils/math.ts',
    }

    # server.js uses CommonJS require against TS siblings.
    assert 'src/utils/math.ts' in graph['server.js']


def test_import_graph_ignores_external_packages(tmp_path: Path) -> None:
    _make_ts_project(tmp_path)

    idx = ASTIndexer()
    index = idx.index_project(tmp_path, [])
    graph = idx.build_import_graph(index)

    # `react` and `express` are bare specifiers and do not become graph edges.
    for src, deps in graph.items():
        for d in deps:
            assert not d.startswith('react')
            assert d != 'express'


def test_call_graph_resolves_cross_file_calls(tmp_path: Path) -> None:
    _make_ts_project(tmp_path)

    idx = ASTIndexer()
    index = idx.index_project(tmp_path, [])
    graph = idx.build_call_graph(index)

    # formatSum calls add which lives in math.ts
    assert 'src/utils/math.ts:add' in graph['src/utils/format.ts:formatSum']

    # Calculator.compute calls add AND multiply across the file boundary,
    # and also calls Calculator.combine (intra-class via `this.`).
    compute_callees = graph['src/components/Calculator.tsx:Calculator.compute']
    assert 'src/utils/math.ts:add' in compute_callees
    assert 'src/utils/math.ts:multiply' in compute_callees
    assert 'src/components/Calculator.tsx:Calculator.combine' in compute_callees

    # server.js handler calls add (destructured from require).
    assert 'src/utils/math.ts:add' in graph['server.js:handler']


def test_dependents_of_ts_file(tmp_path: Path) -> None:
    _make_ts_project(tmp_path)

    idx = ASTIndexer()
    index = idx.index_project(tmp_path, [])
    graph = idx.build_import_graph(index)

    dependents = idx.dependents_of(graph, 'src/utils/math.ts')
    assert dependents == [
        'server.js',
        'src/components/Calculator.tsx',
        'src/index.ts',
        'src/utils/format.ts',
    ]


def test_callers_of_ts_function(tmp_path: Path) -> None:
    _make_ts_project(tmp_path)

    idx = ASTIndexer()
    index = idx.index_project(tmp_path, [])
    graph = idx.build_call_graph(index)

    callers = idx.callers_of(graph, 'src/utils/math.ts:add')
    assert callers == [
        'server.js:handler',
        'src/components/Calculator.tsx:Calculator.compute',
        'src/utils/format.ts:formatSum',
    ]


# ---------------------------------------------------------------------------
# Python parity — the critical invariant
# ---------------------------------------------------------------------------


def test_python_and_ts_can_coexist_in_same_project(tmp_path: Path) -> None:
    _make_ts_project(tmp_path)
    _write(tmp_path, 'helper.py', """
        def greet(name):
            return f"Hello, {name}"

        class Greeter:
            def greet(self, name):
                return f"Hi {name}"
    """)

    idx = ASTIndexer()
    index = idx.index_project(tmp_path, [])

    # Both sets of files are present, each parsed by the correct backend.
    assert 'helper.py' in index['files']
    assert 'src/utils/math.ts' in index['files']

    py = index['files']['helper.py']
    assert {f['name'] for f in py['functions']} == {'greet', 'greet'} or \
           len(py['functions']) == 2  # greet + Greeter.greet
    assert len(py['classes']) == 1 and py['classes'][0]['name'] == 'Greeter'


def test_python_parsing_unchanged_by_ts_extension(tmp_path: Path) -> None:
    """Small Python file → same records as stdlib ast-based parsing would
    produce. This locks in that the dispatch did not regress Python."""
    _write(tmp_path, 'm.py', """
        from os import path

        def top():
            return helper()

        def helper():
            return 1

        class C:
            def method(self):
                return self.top()
    """)

    idx = ASTIndexer()
    index = idx.index_project(tmp_path, [])
    data = index['files']['m.py']

    fn_names = sorted(f['qualified_name'] for f in data['functions'])
    assert fn_names == ['C.method', 'helper', 'top']
    assert [c['name'] for c in data['classes']] == ['C']
    assert data['imports'][0]['module'] == 'os'
    assert data['imports'][0]['names'] == ['path']

    # Call attribution — `top` calls `helper`, `C.method` calls `self.top`.
    edges = {(c['caller_qualified_name'], c['callee']) for c in data['calls']}
    assert ('top', 'helper') in edges
    assert ('C.method', 'self.top') in edges


def test_python_only_project_unchanged_behavior(tmp_path: Path) -> None:
    """A project with no TS/JS files at all must behave exactly like before —
    build_import_graph / build_call_graph should produce the same shape."""
    _write(tmp_path, 'a.py', """
        from b import thing

        def main():
            return thing()
    """)
    _write(tmp_path, 'b.py', """
        def thing():
            return 42
    """)

    idx = ASTIndexer()
    index = idx.index_project(tmp_path, [])

    imports = idx.build_import_graph(index)
    calls = idx.build_call_graph(index)

    assert imports == {'a.py': ['b.py'], 'b.py': []}
    assert calls == {'a.py:main': ['b.py:thing']}
