"""inference.py — text in, text out, structured like HF's TextGenerationPipeline.

Same architecture as the reference (text_generation.py):
    pipeline(text) == __call__ -> preprocess -> _forward -> postprocess
Each stage passes a dict to the next, exactly like the real pipeline. Only the
tokenizer is a toy (char-level); swap in the real GLM tokenizer JSON later and
nothing else changes.
"""
import torch

from modeling_glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM   # YOUR classes


class CharTokenizer:
    """Toy stand-in for the real tokenizer (tokenizer.json). Same encode/decode contract.
    vocab_size=96 = printable ASCII (chars 32..127) shifted to ids 0..95."""

    def __call__(self, text, return_tensors="pt"):            # HF tokenizers are callable
        ids = torch.tensor([[ord(ch) - 32 for ch in text]])   # (1, seq)
        return {"input_ids": ids}

    def decode(self, token_ids):
        return "".join(chr(i + 32) for i in token_ids)


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

        generated_sequence = self.generate(input_ids, **generate_kwargs)

        return {
            "generated_sequence": generated_sequence,
            "input_ids": input_ids,
            "prompt_text": prompt_text,
        }

    def generate(self, input_ids, max_new_tokens=20):
        """The autoregressive loop (the real pipeline calls model.generate for this):
        forward -> argmax last token -> append -> repeat."""
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.model(input_ids)                 # (1, seq, vocab)
            next_token = logits[0, -1].argmax()                # greedy decoding
            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
        return input_ids

    def postprocess(self, model_outputs):
        """ids -> text (reference: line 432). Decode only the NEW tokens, then
        prepend the original prompt (ReturnType.FULL_TEXT behavior)."""
        generated_sequence = model_outputs["generated_sequence"][0].tolist()
        prompt_token_length = model_outputs["input_ids"].shape[-1]
        prompt_text = model_outputs["prompt_text"]

        new_text = self.tokenizer.decode(generated_sequence[prompt_token_length:])
        return [{"generated_text": prompt_text + new_text}]


if __name__ == "__main__":
    config = GlmMoeDsaConfig(vocab_size=96, hidden_size=128, num_hidden_layers=2,
                             num_attention_heads=4, intermediate_size=256)   # toy sizes
    config._attn_implementation = "eager"
    model = GlmMoeDsaForCausalLM(config)
    model.eval()

    pipe = TextGenerationPipeline(model=model, tokenizer=CharTokenizer())
    result = pipe("hello", max_new_tokens=20)
    print(result)          # gibberish after the prompt — weights are random, untrained
