lightweight_cpg.py (integrated)

"""
lightweight_cpg.py
Replaces the Joern/JVM-based CPG pipeline with a pure-Python implementation
using Python's built-in `ast` module and `networkx`.

Outputs the same JSON structure expected by embedder.py and gnn_model_v2.py,
mapped against the node labels defined in vocab.json.
"""

import ast
import networkx as nx
from typing import Any

# ---------------------------------------------------------------------------
# Node label vocabulary (must match vocab.json exactly)
# ---------------------------------------------------------------------------
LABEL_CALL              = "CALL"
LABEL_IDENTIFIER        = "IDENTIFIER"
LABEL_LITERAL           = "LITERAL"
LABEL_PARAM             = "METHOD_PARAMETER_IN"
LABEL_CONTROL           = "CONTROL_STRUCTURE"
LABEL_FUNCTION          = "METHOD"
LABEL_RETURN            = "RETURN"
LABEL_LOCAL             = "LOCAL"
LABEL_IMPORT            = "IMPORT"
LABEL_MODULE            = "MODULE"
LABEL_FILE              = "FILE"
LABEL_NAMESPACE         = "BLOCK"
LABEL_UNKNOWN           = "UNKNOWN"
LABEL_TYPE_DECL         = "TYPE_DECL"
LABEL_COMMENT           = "COMMENT"
LABEL_MEMBER            = "IDENTIFIER"

# Edge labels
EDGE_AST                = "AST"
EDGE_REACHES            = "REACHES"
EDGE_CALL               = "CALL"

# Returned when CPG generation fails (SyntaxError, etc.)
ERROR_CPG: dict = {
    "nodes": [{"id": 0, "code": "PARSE_ERROR", "label": LABEL_UNKNOWN}],
    "edges": []
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _node_code(node: ast.AST) -> str:
    """Return a short, human-readable code snippet for an AST node."""
    if isinstance(node, ast.FunctionDef):
        return f"def {node.name}"
    if isinstance(node, ast.AsyncFunctionDef):
        return f"async def {node.name}"
    if isinstance(node, ast.ClassDef):
        return node.name
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            val = getattr(func.value, 'id', None) or ast.dump(func.value)
            return f"{val}.{func.attr}"
        if isinstance(func, ast.Name):
            return func.id
        return ast.dump(func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Import):
        names = ", ".join(a.name for a in node.names)
        return f"import {names}"
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        names = ", ".join(a.name for a in node.names)
        return f"from {module} import {names}"
    if isinstance(node, (ast.If, ast.For, ast.While, ast.With,
                          ast.AsyncFor, ast.AsyncWith, ast.Try)):
        return type(node).__name__.upper()
    if isinstance(node, ast.Return):
        return "return"
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, ast.Assign):
        targets = ", ".join(
            t.id for t in node.targets if isinstance(t, ast.Name)
        )
        return f"{targets} ="
    if isinstance(node, ast.AugAssign):
        if isinstance(node.target, ast.Name):
            return f"{node.target.id} {type(node.op).__name__}="
    if isinstance(node, ast.Attribute):
        val = getattr(node.value, 'id', None) or "?"
        return f"{val}.{node.attr}"
    return type(node).__name__


def _ast_label(node: ast.AST) -> str:
    """Map an AST node to a vocab.json label."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return LABEL_FUNCTION
    if isinstance(node, ast.ClassDef):
        return LABEL_TYPE_DECL
    if isinstance(node, ast.Call):
        return LABEL_CALL
    if isinstance(node, ast.Name):
        return LABEL_IDENTIFIER
    if isinstance(node, ast.Constant):
        return LABEL_LITERAL
    if isinstance(node, ast.arg):
        return LABEL_PARAM
    if isinstance(node, (ast.If, ast.For, ast.While, ast.With,
                          ast.AsyncFor, ast.AsyncWith, ast.Try,
                          ast.ExceptHandler)):
        return LABEL_CONTROL
    if isinstance(node, ast.Return):
        return LABEL_RETURN
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return LABEL_LOCAL
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return LABEL_IMPORT
    if isinstance(node, ast.Module):
        return LABEL_MODULE
    if isinstance(node, ast.Attribute):
        return LABEL_MEMBER
    return LABEL_UNKNOWN

# ---------------------------------------------------------------------------
# Main CPG builder
# ---------------------------------------------------------------------------

class _CPGBuilder(ast.NodeVisitor):
    """
    Single-pass AST visitor that builds a networkx DiGraph representing
    the Code Property Graph.
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._id_counter: int = 0
        self._parent_stack: list[tuple[int, Any]] = []
        self._ast_to_graph: dict[int, int] = {}
        self._definitions: dict[str, list[int]] = {}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _new_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _add_node(self, ast_node: ast.AST) -> int:
        node_id = self._new_id()
        self.graph.add_node(
            node_id,
            code=_node_code(ast_node),
            label=_ast_label(ast_node),
        )
        self._ast_to_graph[id(ast_node)] = node_id
        return node_id

    def _add_ast_edge(self, parent_gid: int, child_gid: int) -> None:
        self.graph.add_edge(parent_gid, child_gid, label=EDGE_AST)

    def _add_reaches_edge(self, src_gid: int, dst_gid: int) -> None:
        self.graph.add_edge(src_gid, dst_gid, label=EDGE_REACHES)

    def _record_definition(self, name: str, gid: int) -> None:
        self._definitions.setdefault(name, []).append(gid)

    def _lookup_definition(self, name: str) -> int | None:
        defs = self._definitions.get(name)
        return defs[-1] if defs else None

    # ------------------------------------------------------------------
    # Core visitor
    # ------------------------------------------------------------------

    def generic_visit(self, node: ast.AST) -> None:
        """Visit every node, build graph entries, wire AST edges."""
        gid = self._add_node(node)

        if self._parent_stack:
            parent_gid, _ = self._parent_stack[-1]
            self._add_ast_edge(parent_gid, gid)

        self._parent_stack.append((gid, node))
        super().generic_visit(node)
        self._parent_stack.pop()

    # ------------------------------------------------------------------
    # Specialised visitors that add REACHES / data-flow edges
    # ------------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        """Wire: assignment node → identifier nodes on the LHS."""
        self.generic_visit(node)
        assign_gid = self._ast_to_graph[id(node)]
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._record_definition(target.id, assign_gid)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            gid = self._ast_to_graph[id(node)]
            self._record_definition(node.target.id, gid)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            gid = self._ast_to_graph[id(node)]
            self._record_definition(node.target.id, gid)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.generic_visit(node)
        fn_gid = self._ast_to_graph[id(node)]
        self._record_definition(node.name, fn_gid)
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            arg_gid = self._ast_to_graph.get(id(arg))
            if arg_gid is not None:
                self._record_definition(arg.arg, arg_gid)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Name(self, node: ast.Name) -> None:
        """For Name loads, wire a REACHES edge from the definition site."""
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            def_gid = self._lookup_definition(node.id)
            use_gid = self._ast_to_graph[id(node)]
            if def_gid is not None and def_gid != use_gid:
                self._add_reaches_edge(def_gid, use_gid)

    def visit_Call(self, node: ast.Call) -> None:
        """For calls to known names, add a CALL edge from the function def."""
        self.generic_visit(node)
        call_gid = self._ast_to_graph[id(node)]
        callee_name: str | None = None
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee_name = node.func.attr
        if callee_name:
            def_gid = self._lookup_definition(callee_name)
            if def_gid is not None:
                self.graph.add_edge(def_gid, call_gid, label=EDGE_CALL)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_cpg(code_string: str) -> dict:
    """
    Parse *code_string* as Python source and return a CPG dictionary.

    The returned structure is::

        {
            "nodes": [{"id": <int>, "code": <str>, "label": <str>}, ...],
            "edges": [{"src": <int>, "dst": <int>, "label": <str>}, ...]
        }

    On ``SyntaxError`` or any other parse failure the function returns
    :data:`ERROR_CPG` so callers never receive an exception.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return ERROR_CPG
    except Exception:
        return ERROR_CPG

    builder = _CPGBuilder()
    builder.visit(tree)

    g = builder.graph

    nodes = [
        {
            "id": nid,
            "code": data.get("code", ""),
            "label": data.get("label", LABEL_UNKNOWN),
        }
        for nid, data in g.nodes(data=True)
    ]

    edges = [
        {
            "src": src,
            "dst": dst,
            "label": data.get("label", EDGE_AST),
        }
        for src, dst, data in g.edges(data=True)
    ]

    if not nodes:
        nodes = [{"id": 1, "code": code_string[:80], "label": LABEL_MODULE}]

    return {"nodes": nodes, "edges": edges}
