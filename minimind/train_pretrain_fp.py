"""train_pretrain_fp.py — minimind's train_pretrain.py translated into FUNCTIONAL PROGRAMMING.

Same logic, opposite philosophy. The mutating original:
    optimizer.step()                    # weights change in place; returns nothing
becomes:
    state, metrics = train_step(state, batch, hp)    # world untouched; NEW state returned

The FP rules followed here:
  1. ALL changing things live in one explicit `state` tuple: (params, opt_state, step).
  2. Every step function is PURE: state in -> new state out. Old state is never modified.
  3. The model runs statelessly via torch.func.functional_call(model, params, ...) —
     the nn.Module is just a frozen description; weights travel as an ordinary dict.
  4. AdamW is hand-written as a pure function (PyTorch's optim.AdamW is a mutator, so
     it cannot be used) — its momentum memory is carried openly in opt_state.
  5. Side effects (print, save, data loading) are quarantined at the program edges,
     never inside the pure core.

Stripped, as in david/training/train.py: DDP, wandb, resume, autocast/GradScaler
(mutation-era armor; JAX-style code gets these back via jit/pmap instead).
"""
import argparse
import math
import os

import torch
from torch.func import functional_call, grad
from torch.utils.data import DataLoader

from lm_dataset import PretrainDataset  # noqa: F401  (flat copy; original: dataset.lm_dataset)
from model_minimind import MiniMindConfig, MiniMindForCausalLM  # (original: model.model_minimind)

# =====================================================================================
# PURE CORE — no mutation, no side effects below this line
# =====================================================================================


def get_lr(current_step, total_steps, lr):
    """Already pure in the original (trainer_utils.py:40) — copied verbatim."""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def compute_loss(params, model, input_ids, labels):
    """Pure: (weights, frozen model description, batch) -> scalar loss.
    Mirrors the original's `res = model(input_ids, labels=labels); res.loss + res.aux_loss`,
    but the weights are an ARGUMENT, not hidden state inside `model`."""
    res = functional_call(model, params, (input_ids,), {"labels": labels})
    return res.loss + res.aux_loss


def adamw_update(params, grads, opt_state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """Pure AdamW: returns NEW params and NEW optimizer memory; touches neither input.
    This is exactly what optim.AdamW.step() does by mutation, written as math."""
    exp_avg, exp_avg_sq, t = opt_state
    t = t + 1
    new_params, new_avg, new_avg_sq = {}, {}, {}
    for k in params:
        g = grads[k]
        m = betas[0] * exp_avg[k] + (1 - betas[0]) * g            # momentum (new tensor)
        v = betas[1] * exp_avg_sq[k] + (1 - betas[1]) * g * g     # variance (new tensor)
        m_hat = m / (1 - betas[0] ** t)                           # bias correction
        v_hat = v / (1 - betas[1] ** t)
        p = params[k] * (1 - lr * weight_decay)                   # decoupled weight decay
        new_params[k] = p - lr * m_hat / (v_hat.sqrt() + eps)
        new_avg[k], new_avg_sq[k] = m, v
    return new_params, (new_avg, new_avg_sq, t)


def clip_grads(grads, max_norm):
    """Pure version of torch.nn.utils.clip_grad_norm_ (which mutates .grad in place)."""
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads.values()))
    scale = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    return {k: g * scale for k, g in grads.items()}, total_norm


def train_step(state, batch, model, hp):
    """ONE training step, pure: state in -> (new state, metrics) out.
    The original's train_epoch body (lr -> forward -> backward -> clip -> step),
    with every mutation replaced by a returned value."""
    params, opt_state, step = state
    input_ids, labels = batch

    lr = get_lr(step, hp["total_steps"], hp["learning_rate"])     # was: mutate param_group["lr"]
    loss, grads = _loss_and_grads(params, model, input_ids, labels)  # was: loss.backward() filling .grad
    grads, grad_norm = clip_grads(grads, hp["grad_clip"])         # was: clip_grad_norm_ in-place
    params, opt_state = adamw_update(params, grads, opt_state, lr)  # was: optimizer.step()
    # (no zero_grad needed: grads were never stored anywhere to clear)

    return (params, opt_state, step + 1), {"loss": loss, "lr": lr, "grad_norm": grad_norm}


def _loss_and_grads(params, model, input_ids, labels):
    """grad() transforms compute_loss into a NEW FUNCTION returning gradients —
    differentiation as a function transformation (the JAX idea), not a side effect."""
    loss = compute_loss(params, model, input_ids, labels)
    grads = grad(compute_loss)(params, model, input_ids, labels)
    return loss.detach(), grads


def init_state(model):
    """Build state0: weights lifted OUT of the module into a plain dict, plus
    AdamW's empty memory. From here on, the module never changes again."""
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    opt_state = (
        {k: torch.zeros_like(v) for k, v in params.items()},     # exp_avg   (momentum)
        {k: torch.zeros_like(v) for k, v in params.items()},     # exp_avg_sq (variance)
        0,                                                        # t
    )
    return (params, opt_state, 1)


# =====================================================================================
# EDGES — side effects allowed only here (I/O, printing, saving), mirroring the
# original's __main__ sections
# =====================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining (functional style)")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=340, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl")
    args = parser.parse_args()

    # ========== 1. seed ==========
    torch.manual_seed(42)

    # ========== 2. dirs + config ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size,
                               num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe))

    # ========== 5. model (frozen description), data, and state0 (replaces optimizer) ====
    model = MiniMindForCausalLM(lm_config).to(args.device)
    model.train()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model")
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    state = init_state(model)                       # ALL mutable-ness of training, in one tuple
    hp = {"learning_rate": args.learning_rate, "grad_clip": args.grad_clip,
          "total_steps": args.epochs * len(loader)}

    # ========== 8. train: the loop is a FOLD over batches ==========
    for epoch in range(args.epochs):
        for input_ids, labels in loader:
            batch = (input_ids.to(args.device), labels.to(args.device))

            state, metrics = train_step(state, batch, model, hp)   # the ONLY line that "changes" anything —
            #                                                        and even it just rebinds a name

            params, opt_state, step = state
            if step % args.log_interval == 0:                       # edge: printing
                print(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{hp['total_steps']}), "
                      f"loss: {metrics['loss'].item():.4f}, lr: {metrics['lr']:.8f}")
            if step % args.save_interval == 0:                      # edge: checkpoint = state written down
                ckp = f"{args.save_dir}/pretrain_fp_{lm_config.hidden_size}.pth"
                torch.save({k: v.half().cpu() for k, v in params.items()}, ckp)
