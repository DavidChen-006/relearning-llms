"""A/B: same config class, with vs without @strict.

Run:
    python david/demo/demo_strict.py
"""

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig


class ConfigA(PreTrainedConfig):
    """WITHOUT @strict — loose validation."""

    model_type = "glm_moe_dsa"
    hidden_size: int = 128
    num_hidden_layers: int = 4


@strict
class ConfigB(PreTrainedConfig):
    """WITH @strict — type-checked on init and assignment."""

    model_type = "glm_moe_dsa"
    hidden_size: int = 128
    num_hidden_layers: int = 4


def _try(label: str, fn) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {label}")
    print(f"{'─' * 72}")
    try:
        result = fn()
        print(f"  OK → {result!r}")
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")


def main() -> None:
    print("A/B: @strict on PreTrainedConfig subclasses")
    print("Same fields — only the decorator differs.\n")

    _try("A  valid init", lambda: ConfigA(hidden_size=128))
    _try("B  valid init", lambda: ConfigB(hidden_size=128))

    _try(
        "A  bad type at init: hidden_size='oops'",
        lambda: ConfigA(hidden_size="oops"),
    )
    _try(
        "B  bad type at init: hidden_size='oops'",
        lambda: ConfigB(hidden_size="oops"),
    )

    def assign_a():
        c = ConfigA(hidden_size=128)
        c.hidden_size = "oops"
        return c.hidden_size

    def assign_b():
        c = ConfigB(hidden_size=128)
        c.hidden_size = "oops"
        return c.hidden_size

    _try("A  bad assignment: config.hidden_size = 'oops'", assign_a)
    _try("B  bad assignment: config.hidden_size = 'oops'", assign_b)

    print(f"\n{'─' * 72}")
    print("  @strict catches wrong types at init AND on assignment.")
    print("  Without it, hidden_size='oops' silently sticks — then breaks later in the model.")
    print(f"{'─' * 72}")


if __name__ == "__main__":
    main()
