"""inference.py — text in, text out, structured like HF's TextGenerationPipeline.

Same architecture as the reference (text_generation.py):
    pipeline(text) == __call__ -> preprocess -> _forward -> postprocess
Each stage passes a dict to the next, exactly like the real pipeline. Only the
tokenizer is a toy (char-level); swap in the real GLM tokenizer JSON later and
nothing else changes.
"""
import argparse
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture"))
from modeling_glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM   # YOUR classes  # noqa: E402


class TextGenerationPipeline:
    """Mirror of transformers' TextGenerationPipeline, minus the production armor."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, prompt_text, max_new_tokens=20):
        model_inputs = self.preprocess(prompt_text)
        model_outputs = self._forward(model_inputs, max_new_tokens=max_new_tokens)
        return self.postprocess(model_outputs)

    def preprocess(self, prompt_text):
        """text -> ids (reference: line 301). Tokenize, then carry the prompt along."""
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs["prompt_text"] = prompt_text
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        """ids -> ids (reference: line 369). Hand everything to model.generate()."""
        input_ids = model_inputs["input_ids"]
        prompt_text = model_inputs.pop("prompt_text")

        generated_sequence = self.model.generate(              # reference line 403
            input_ids,
            eos_token_id=self.tokenizer.eos_token_id,          # stop when the model says "I'm done" (id 2)
            **generate_kwargs,
        )

        return {
            "generated_sequence": generated_sequence,
            "input_ids": input_ids,
            "prompt_text": prompt_text,
        }

    def postprocess(self, model_outputs):
        """ids -> text (reference: line 432). Decode only the NEW tokens, then
        prepend the original prompt (ReturnType.FULL_TEXT behavior)."""
        generated_sequence = model_outputs["generated_sequence"][0].tolist()
        prompt_token_length = model_outputs["input_ids"].shape[-1]
        prompt_text = model_outputs["prompt_text"]

        new_text = self.tokenizer.decode(generated_sequence[prompt_token_length:])
        return [{"generated_text": prompt_text + new_text}]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text generation with the toy GLM model")
    parser.add_argument("--prompt", help="one-shot prompt; omit for interactive mode")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    # expensive one-time setup: build model + tokenizer + pipeline ONCE
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(os.path.abspath(__file__)))   # tokenizer.json (minimind's 6400-vocab BPE)
    config = GlmMoeDsaConfig(vocab_size=tokenizer.vocab_size, hidden_size=128, num_hidden_layers=2,
                             num_attention_heads=4, intermediate_size=256)   # toy sizes
    config._attn_implementation = "eager"
    model = GlmMoeDsaForCausalLM(config)
    model.eval()
    pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    # cheap repeated call: one-shot (CLI) or loop (REPL) — output is gibberish until trained
    if args.prompt is not None:
        print(pipe(args.prompt, max_new_tokens=args.max_new_tokens))
    else:
        print("interactive mode — Ctrl-C or empty line to quit")
        while True:
            prompt = input("> ")
            if not prompt:
                break
            print(pipe(prompt, max_new_tokens=args.max_new_tokens))
