"""A/B: same config class, with vs without @auto_docstring.

Run:
    python david/demo/demo_auto_docstring.py
"""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import auto_docstring

@strict
class ConfigA(PreTrainedConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of top tokens selected by the indexer for sparse attention.
    """

    model_type = "glm_moe_dsa"
    hidden_size: int = 128
    num_hidden_layers: int = 4
    n_group: int = 1
    index_topk: int = 64


@auto_docstring(checkpoint="zai-org/GLM-5", custom_intro="GLM MoE DSA configuration.")
@strict
class ConfigB(PreTrainedConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of top tokens selected by the indexer for sparse attention.
    """

    model_type = "glm_moe_dsa"
    hidden_size: int = 128
    num_hidden_layers: int = 4
    n_group: int = 1
    index_topk: int = 64


def _print_side(label: str, doc: str | None) -> tuple[int, int]:
    text = doc or ""
    lines = text.count("\n") + (1 if text else 0)
    print(f"\n{'─' * 72}")
    print(f"  {label}  ({len(text)} chars, {lines} lines)")
    print(f"{'─' * 72}\n")
    print(text)
    return len(text), lines


def main() -> None:
    print("A/B: @auto_docstring(checkpoint='zai-org/GLM-5')")
    print("Same class, same fields, same hand-written doc — decorator is the only difference.\n")

    chars_a, lines_a = _print_side("A  WITHOUT @auto_docstring", ConfigA.__doc__)
    chars_b, lines_b = _print_side("B  WITH    @auto_docstring", ConfigB.__doc__)

    print(f"\n{'─' * 72}")
    print(f"  Δ  +{chars_b - chars_a} chars, +{lines_b - lines_a} lines")
    print(f"  Runtime identical: ConfigA().hidden_size == ConfigB().hidden_size == {ConfigA().hidden_size}")
    print(f"{'─' * 72}")


if __name__ == "__main__":
    main()
