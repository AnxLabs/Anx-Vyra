"""Flag names read at top-level flow before any binding.

Walks statements in source order, through if/try/for/while/with bodies but
not into function or class bodies (those run later). A Name load in that
flow must be a builtin or bound by an earlier statement. Conservative: a
binding in any earlier branch counts. Catches "used before import" in the
packaged treatment, which killed the expanded40 Kaggle run.
"""
import ast
import builtins

_SKIP = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _flow_names(node):
    """Yield (kind, name, lineno) for a statement, excluding nested scopes."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for d in n.decorator_list:
                stack.append(d)
            yield "store", n.name, n.lineno
            continue
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # only the first iterable is evaluated in the enclosing scope
            stack.append(n.generators[0].iter)
            continue
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                yield "store", (a.asname or a.name).split(".")[0], n.lineno
            continue
        if isinstance(n, ast.ExceptHandler) and n.name:
            yield "store", n.name, n.lineno
        if isinstance(n, ast.Name):
            yield ("load" if isinstance(n.ctx, ast.Load) else "store"), n.id, n.lineno
        stack.extend(ast.iter_child_nodes(n))


def _ordered(stmts):
    for s in stmts:
        if isinstance(s, (ast.If, ast.While, ast.For)):
            yield from _flow_part(s.test if not isinstance(s, ast.For) else s.iter)
            if isinstance(s, ast.For):
                yield from _flow_part(s.target)
            yield from _ordered(s.body)
            yield from _ordered(s.orelse)
        elif isinstance(s, ast.Try):
            yield from _ordered(s.body)
            for h in s.handlers:
                if h.name:
                    yield "store", h.name, h.lineno
                yield from _ordered(h.body)
            yield from _ordered(s.orelse)
            yield from _ordered(s.finalbody)
        elif isinstance(s, ast.With):
            for item in s.items:
                yield from _flow_part(item.context_expr)
                if item.optional_vars is not None:
                    yield from _flow_part(item.optional_vars)
            yield from _ordered(s.body)
        else:
            # value side before target side for assignments
            if isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and s.value is not None:
                yield from _flow_part(s.value)
                targets = s.targets if isinstance(s, ast.Assign) else [s.target]
                for t in targets:
                    yield from _flow_part(t)
            else:
                yield from _flow_part(s)


def _flow_part(node):
    loads, stores = [], []
    for kind, name, line in _flow_names(node):
        (loads if kind == "load" else stores).append((kind, name, line))
    yield from loads
    yield from stores


def used_before_binding(source, predefined=()):
    """Return [(name, lineno)] for top-level-flow loads with no earlier binding."""
    bound = set(dir(builtins)) | {"__file__", "__name__"} | set(predefined)
    bad = []
    for kind, name, line in _ordered(ast.parse(source).body):
        if kind == "store":
            bound.add(name)
        elif name not in bound:
            bad.append((name, line))
    return bad
