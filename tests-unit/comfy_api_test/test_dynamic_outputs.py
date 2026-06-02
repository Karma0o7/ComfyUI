"""Unit tests for ``DynamicOutputs.ByKey`` and the finalized-outputs path."""

import pytest

from comfy_api.latest import _io as io


# ---------------------------------------------------------------------------
# Schema-level construction and validation
# ---------------------------------------------------------------------------

def _byke():
    return io.DynamicOutputs.ByKey(
        id="result",
        selector="mode",
        options=[
            io.DynamicOutputs.Option(key="image",
                                     outputs=[io.Image.Output("image"), io.Mask.Output("mask")]),
            io.DynamicOutputs.Option(key="latent",
                                     outputs=[io.Latent.Output("latent")]),
        ],
    )


def test_option_rejects_empty_key():
    with pytest.raises(ValueError, match="non-empty string"):
        io.DynamicOutputs.Option(key="", outputs=[])


def test_option_rejects_non_output_entry():
    with pytest.raises(ValueError, match="Output instances"):
        io.DynamicOutputs.Option(key="x", outputs=["not an output"])


def test_option_requires_explicit_output_ids():
    with pytest.raises(ValueError, match="declare an id"):
        io.DynamicOutputs.Option(key="x", outputs=[io.Image.Output()])  # no id


def test_bykey_rejects_empty_options():
    with pytest.raises(ValueError, match="at least one Option"):
        io.DynamicOutputs.ByKey(id="r", selector="m", options=[])


def test_bykey_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate option key"):
        io.DynamicOutputs.ByKey(
            id="r", selector="m",
            options=[
                io.DynamicOutputs.Option(key="x", outputs=[io.Image.Output("a")]),
                io.DynamicOutputs.Option(key="x", outputs=[io.Latent.Output("b")]),
            ],
        )


def test_bykey_rejects_duplicate_output_ids_across_options():
    with pytest.raises(ValueError, match="appears in more than one option"):
        io.DynamicOutputs.ByKey(
            id="r", selector="m",
            options=[
                io.DynamicOutputs.Option(key="x", outputs=[io.Image.Output("dup")]),
                io.DynamicOutputs.Option(key="y", outputs=[io.Latent.Output("dup")]),
            ],
        )


# ---------------------------------------------------------------------------
# Schema integration
# ---------------------------------------------------------------------------

def _make_node(extra_outputs=None):
    """Build a V3 node class with a selector input + DynamicOutputs group."""
    extras = extra_outputs or []

    class DynNode(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="DynNode",
                inputs=[io.Combo.Input("mode", options=["image", "latent"], default="image")],
                outputs=[*extras, _byke()],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({"image": None, "mask": None})

    return DynNode


def test_schema_validate_rejects_unknown_selector():
    class BadSelector(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="BadSelector",
                inputs=[io.Combo.Input("not_mode", options=["a"])],
                outputs=[
                    io.DynamicOutputs.ByKey(
                        id="r", selector="mode",
                        options=[io.DynamicOutputs.Option(key="a", outputs=[io.Image.Output("a")])],
                    ),
                ],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({"a": None})

    with pytest.raises(ValueError, match="selector input 'mode' does not exist"):
        BadSelector.GET_SCHEMA()


def test_schema_validate_rejects_id_collision_with_static_output():
    class Collision(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="Collision",
                inputs=[io.Combo.Input("mode", options=["a"])],
                outputs=[
                    io.Image.Output("shared"),
                    io.DynamicOutputs.ByKey(
                        id="r", selector="mode",
                        options=[io.DynamicOutputs.Option(key="a", outputs=[io.Latent.Output("shared")])],
                    ),
                ],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({"shared": None})

    with pytest.raises(ValueError, match="Output ids must be unique"):
        Collision.GET_SCHEMA()


def test_schema_get_v1_info_emits_dynamic_outputs_field():
    DynNode = _make_node()
    DynNode.GET_SCHEMA()
    info = DynNode.SCHEMA.get_v1_info(DynNode)
    assert info.dynamic_outputs is not None and len(info.dynamic_outputs) == 1
    group = info.dynamic_outputs[0]
    assert group["kind"] == "by_key"
    assert group["selector"] == "mode"
    assert {opt["key"] for opt in group["options"]} == {"image", "latent"}
    # Static output arrays are empty — only the dynamic group is declared.
    assert info.output == []
    assert info.output_is_list == []


def test_schema_static_outputs_stable_prefix_in_v1_arrays():
    """A static output before a dynamic group still surfaces in RETURN_TYPES etc."""
    DynNode = _make_node(extra_outputs=[io.String.Output("status")])
    DynNode.GET_SCHEMA()
    # Class-level static arrays are the always-present prefix.
    assert list(DynNode.RETURN_TYPES) == ["STRING"]
    assert list(DynNode.RETURN_NAMES) == ["status"]
    assert list(DynNode.OUTPUT_IS_LIST) == [False]


# ---------------------------------------------------------------------------
# get_finalized_class_outputs
# ---------------------------------------------------------------------------

def test_finalize_picks_active_branch():
    schema_outputs = [_byke()]
    finalized = io.get_finalized_class_outputs(schema_outputs, {"mode": "latent"})
    assert finalized.output_ids == ["latent"]
    assert finalized.return_types == ["LATENT"]
    assert finalized.output_is_list == [False]


def test_finalize_unknown_selector_yields_empty():
    schema_outputs = [_byke()]
    finalized = io.get_finalized_class_outputs(schema_outputs, {"mode": "nonexistent"})
    assert len(finalized) == 0


def test_finalize_link_selector_yields_empty():
    """Link as selector value is treated as 'not finalizable' — no branch."""
    schema_outputs = [_byke()]
    finalized = io.get_finalized_class_outputs(schema_outputs, {"mode": ["src", 0]})
    assert len(finalized) == 0


def test_finalize_static_prefix_preserved():
    schema_outputs = [io.String.Output("status"), _byke()]
    finalized = io.get_finalized_class_outputs(schema_outputs, {"mode": "image"})
    assert finalized.output_ids == ["status", "image", "mask"]
    assert finalized.return_types == ["STRING", "IMAGE", "MASK"]


# ---------------------------------------------------------------------------
# NodeOutput.from_named
# ---------------------------------------------------------------------------

def test_nodeoutput_from_named_stores_dict():
    out = io.NodeOutput.from_named({"a": 1, "b": 2})
    assert out.named == {"a": 1, "b": 2}
    assert out.args == ()
    assert out.result is None  # `.result` is the positional tuple


def test_nodeoutput_rejects_mixed_positional_and_named():
    with pytest.raises(ValueError, match="cannot mix positional"):
        io.NodeOutput(1, 2, named={"a": 1})


# ---------------------------------------------------------------------------
# Group-id uniqueness
# ---------------------------------------------------------------------------

def test_schema_rejects_duplicate_dynamic_group_ids():
    class Dup(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="Dup",
                inputs=[io.Combo.Input("mode", options=["a"])],
                outputs=[
                    io.DynamicOutputs.ByKey(id="r", selector="mode",
                        options=[io.DynamicOutputs.Option(key="a", outputs=[io.Image.Output("x")])]),
                    io.DynamicOutputs.ByKey(id="r", selector="mode",
                        options=[io.DynamicOutputs.Option(key="a", outputs=[io.Latent.Output("y")])]),
                ],
            )

        @classmethod
        def execute(cls, **kwargs):
            return io.NodeOutput.from_named({"x": None, "y": None})

    with pytest.raises(ValueError, match="DynamicOutputs group ids must be unique"):
        Dup.GET_SCHEMA()
