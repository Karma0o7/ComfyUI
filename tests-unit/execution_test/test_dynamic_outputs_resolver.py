"""TypeResolver + execution-helper tests for ``DynamicOutputs.ByKey``.

Covers the wiring between the per-prompt finalized output list and the
execution layer:

  * type resolver returns the active branch's declared type
  * type resolver reports the active output count for stale-link validation
  * ``is_output_list`` reflects the active branch
  * execution helpers refuse to consume ``NodeOutput(named=...)`` against a
    non-dynamic node, and reorder against the finalized list for dynamic ones
"""

from __future__ import annotations

import sys
import types as _pytypes

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures (mirror tests-unit/execution_test/test_type_resolver.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_nodes_module():
    real_nodes = sys.modules.get("nodes")
    fake = _pytypes.ModuleType("nodes")
    fake.NODE_CLASS_MAPPINGS = {}
    sys.modules["nodes"] = fake
    try:
        yield fake.NODE_CLASS_MAPPINGS
    finally:
        if real_nodes is not None:
            sys.modules["nodes"] = real_nodes
        else:
            del sys.modules["nodes"]


@pytest.fixture
def TypeResolver(fake_nodes_module):
    from comfy_execution.type_resolver import TypeResolver as TR
    return TR


def _v1_node(return_types: tuple[str, ...]):
    class _V1:
        RETURN_TYPES = return_types

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    return _V1


def _make_dyn_node():
    """V3 node: ``mode`` selector with two branches."""
    from comfy_api.latest import _io as io

    class DynBranch(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="DynBranch",
                inputs=[io.Combo.Input("mode", options=["image", "latent"], default="image")],
                outputs=[
                    io.DynamicOutputs.ByKey(
                        id="result", selector="mode",
                        options=[
                            io.DynamicOutputs.Option(key="image", outputs=[
                                io.Image.Output("image"),
                                io.Mask.Output("mask"),
                            ]),
                            io.DynamicOutputs.Option(key="latent", outputs=[
                                io.Latent.Output("latent"),
                            ]),
                        ],
                    ),
                ],
            )

        @classmethod
        def execute(cls, mode):
            if mode == "latent":
                return io.NodeOutput.from_named({"latent": None})
            return io.NodeOutput.from_named({"image": None, "mask": None})

    DynBranch.GET_SCHEMA()
    return DynBranch


# ---------------------------------------------------------------------------
# TypeResolver against finalized outputs
# ---------------------------------------------------------------------------

def test_dynamic_resolve_picks_active_branch_image(fake_nodes_module, TypeResolver):
    fake_nodes_module["DynBranch"] = _make_dyn_node()
    prompt = {"n1": {"class_type": "DynBranch", "inputs": {"mode": "image"}}}
    r = TypeResolver(prompt)
    assert r.resolve_output_type("n1", 0) == "IMAGE"
    assert r.resolve_output_type("n1", 1) == "MASK"


def test_dynamic_resolve_picks_active_branch_latent(fake_nodes_module, TypeResolver):
    fake_nodes_module["DynBranch"] = _make_dyn_node()
    prompt = {"n1": {"class_type": "DynBranch", "inputs": {"mode": "latent"}}}
    r = TypeResolver(prompt)
    assert r.resolve_output_type("n1", 0) == "LATENT"


def test_dynamic_finalized_output_count(fake_nodes_module, TypeResolver):
    fake_nodes_module["DynBranch"] = _make_dyn_node()
    fake_nodes_module["Static"] = _v1_node(("INT", "FLOAT"))
    prompt = {
        "img": {"class_type": "DynBranch", "inputs": {"mode": "image"}},
        "lat": {"class_type": "DynBranch", "inputs": {"mode": "latent"}},
        "stat": {"class_type": "Static", "inputs": {}},
    }
    r = TypeResolver(prompt)
    assert r.finalized_output_count("img") == 2  # image + mask
    assert r.finalized_output_count("lat") == 1
    assert r.finalized_output_count("stat") == 2  # static V1 falls through


def test_dynamic_is_output_list_reflects_branch(fake_nodes_module, TypeResolver):
    from comfy_api.latest import _io as io

    class DynList(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="DynList",
                inputs=[io.Combo.Input("mode", options=["one", "many"], default="one")],
                outputs=[
                    io.DynamicOutputs.ByKey(
                        id="r", selector="mode",
                        options=[
                            io.DynamicOutputs.Option(key="one", outputs=[
                                io.Image.Output("img"),
                            ]),
                            io.DynamicOutputs.Option(key="many", outputs=[
                                io.Image.Output("imgs", is_output_list=True),
                            ]),
                        ],
                    ),
                ],
            )

        @classmethod
        def execute(cls, mode):
            return io.NodeOutput.from_named({"img": None} if mode == "one" else {"imgs": [None]})

    DynList.GET_SCHEMA()
    fake_nodes_module["DynList"] = DynList
    prompt = {
        "one": {"class_type": "DynList", "inputs": {"mode": "one"}},
        "many": {"class_type": "DynList", "inputs": {"mode": "many"}},
    }
    r = TypeResolver(prompt)
    assert r.is_output_list("one", 0) is False
    assert r.is_output_list("many", 0) is True


def test_dynamic_out_of_range_returns_any(fake_nodes_module, TypeResolver):
    """Slot index beyond the finalized branch resolves to AnyType (validation rejects separately)."""
    fake_nodes_module["DynBranch"] = _make_dyn_node()
    prompt = {"n1": {"class_type": "DynBranch", "inputs": {"mode": "latent"}}}
    r = TypeResolver(prompt)
    assert r.resolve_output_type("n1", 5) == "*"


# ---------------------------------------------------------------------------
# Execution-side helpers
# ---------------------------------------------------------------------------

def test_normalize_named_result_reorders_to_finalized():
    from comfy_api.latest import _io as io
    from execution import _normalize_named_result

    finalized = io.get_finalized_class_outputs(
        [io.DynamicOutputs.ByKey(
            id="r", selector="mode",
            options=[io.DynamicOutputs.Option(key="x", outputs=[
                io.Image.Output("a"), io.Mask.Output("b"), io.Latent.Output("c"),
            ])],
        )],
        {"mode": "x"},
    )
    node_output = io.NodeOutput.from_named({"c": 30, "a": 10, "b": 20})
    assert _normalize_named_result(node_output, finalized) == (10, 20, 30)


def test_normalize_named_result_rejects_unknown_or_missing_ids():
    from comfy_api.latest import _io as io
    from execution import _normalize_named_result

    finalized = io.get_finalized_class_outputs(
        [io.DynamicOutputs.ByKey(
            id="r", selector="mode",
            options=[io.DynamicOutputs.Option(key="x", outputs=[
                io.Image.Output("a"), io.Mask.Output("b"),
            ])],
        )],
        {"mode": "x"},
    )
    with pytest.raises(Exception, match="missing"):
        _normalize_named_result(io.NodeOutput.from_named({"a": 1}), finalized)
    with pytest.raises(Exception, match="unknown"):
        _normalize_named_result(io.NodeOutput.from_named({"a": 1, "b": 2, "z": 3}), finalized)


def test_normalize_named_result_requires_dynamic_node():
    from comfy_api.latest import _io as io
    from execution import _normalize_named_result

    with pytest.raises(Exception, match="DynamicOutputs"):
        _normalize_named_result(io.NodeOutput.from_named({"a": 1}), None)


# ---------------------------------------------------------------------------
# Blocker / output-shape paths through get_output_from_returns
# ---------------------------------------------------------------------------

def _dyn_finalized(branch_outputs):
    from comfy_api.latest import _io as io
    return io.get_finalized_class_outputs(
        [io.DynamicOutputs.ByKey(id="r", selector="mode", options=[
            io.DynamicOutputs.Option(key="x", outputs=branch_outputs),
        ])],
        {"mode": "x"},
    )


def test_blocker_sized_to_finalized_outputs_for_node_output():
    """V3 node returning a bare ``ExecutionBlocker`` must yield blocker tuples
    sized to the active output count, not the empty static RETURN_TYPES."""
    from comfy_api.latest import _io as io
    from comfy_execution.graph_utils import ExecutionBlocker
    from execution import get_output_from_returns

    finalized = _dyn_finalized([io.Image.Output("a"), io.Mask.Output("b")])

    class _Obj:
        RETURN_TYPES = ()  # only static outputs — dynamic group lives in schema

    out = io.NodeOutput(block_execution="paused")
    output, _ui, has_subgraph = get_output_from_returns([out], _Obj(), finalized_outputs=finalized)
    assert has_subgraph is False
    # merge_result_data flattens per-slot, one input → list-of-one per slot
    assert len(output) == 2
    for slot in output:
        assert len(slot) == 1
        assert isinstance(slot[0], ExecutionBlocker)
        assert slot[0].message == "paused"


def test_bare_execution_blocker_sized_to_finalized_outputs():
    """The non-NodeOutput path (bare ``ExecutionBlocker`` from V1-style returns)
    also sizes against the finalized list."""
    from comfy_api.latest import _io as io
    from comfy_execution.graph_utils import ExecutionBlocker
    from execution import get_output_from_returns

    finalized = _dyn_finalized([io.Image.Output("a"), io.Mask.Output("b"), io.Latent.Output("c")])

    class _Obj:
        RETURN_TYPES = ()

    blocker = ExecutionBlocker("stopped")
    output, _ui, has_subgraph = get_output_from_returns([blocker], _Obj(), finalized_outputs=finalized)
    assert has_subgraph is False
    assert len(output) == 3
    for slot in output:
        assert slot[0] is blocker
