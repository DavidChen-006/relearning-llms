"""demo_adamw_vs_sgd.py — ISOLATE what AdamW adds on top of SGD's naked equation.

Same .step() button, different engine. SGD is w -= lr*grad, literally. AdamW keeps
that skeleton but changes three things, each isolated in its own part:
  PART 1: ADAPTIVITY  — one step, wildly different grads -> SGD steps are proportional
          to the grad; AdamW steps are ~lr for every weight (per-weight normalization)
  PART 2: MOMENTUM    — a zigzag gradient over many steps -> SGD zigzags with it;
          AdamW's running average cancels the noise and drifts in the true direction
  PART 3: WEIGHT DECAY — zero gradient -> SGD never moves; AdamW(weight_decay>0)
          still shrinks the weight toward 0 (the "W": decay decoupled from the grad)
  PART 4: verify — hand-write AdamW's first step and match torch's number exactly

(train.py / minimind line 138 pick AdamW at CREATION: optim.AdamW(...). The loop's
optimizer.step() then runs THIS math instead of SGD's.)
"""
import torch

torch.set_printoptions(precision=4, sci_mode=False)

lr = 0.1

# ================================================================ PART 1
print("=" * 74)
print("PART 1 — ADAPTIVITY: same lr, grads of wildly different sizes, ONE step")
print("=" * 74)

# four weights, four hand-picked gradients: huge, tiny, medium, huge-negative
W0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
G = torch.tensor([100.0, 0.01, 1.0, -100.0])

w_sgd = W0.clone().requires_grad_(True)
w_sgd.grad = G.clone()
opt_sgd = torch.optim.SGD([w_sgd], lr=lr)
opt_sgd.step()

w_adamw = W0.clone().requires_grad_(True)
w_adamw.grad = G.clone()
opt_adamw = torch.optim.AdamW([w_adamw], lr=lr, weight_decay=0.0)  # decay off, isolate adaptivity
opt_adamw.step()

print(f"{'grad':>10} {'SGD change':>12} {'AdamW change':>13}")
for g, s, a in zip(G, w_sgd.data - W0, w_adamw.data - W0):
    print(f"{g.item():>10.2f} {s.item():>+12.4f} {a.item():>+13.4f}")
print("""
 -> SGD's step = -lr*grad: the grad=100 weight jumped 10.0, the grad=0.01 weight
    barely moved 0.001. A 10000x gradient gap becomes a 10000x step gap.
 -> AdamW divides each weight's step by that weight's own gradient size
    (sqrt of the running average of grad^2). Every step lands near lr=0.1,
    only the SIGN survives from the gradient. No weight can dominate or stall
    just because its gradient happens to be huge or tiny.""")

# ================================================================ PART 2
print("=" * 74)
print("PART 2 — MOMENTUM: a zigzag gradient (+5, -4.9, +5, -4.9, ...) for 40 steps")
print("=" * 74)

# the noisy-batch situation: gradient flips sign every step, but its AVERAGE is +0.05
# (the true downhill direction). watch who finds it.
w_sgd = torch.tensor([0.0], requires_grad=True)
w_adamw = torch.tensor([0.0], requires_grad=True)
opt_sgd = torch.optim.SGD([w_sgd], lr=0.01)
opt_adamw = torch.optim.AdamW([w_adamw], lr=0.01, weight_decay=0.0)

print(f"{'step':>5} {'grad':>7} {'SGD w':>9} {'AdamW w':>9}")
for step in range(1, 41):
    g = 5.0 if step % 2 == 1 else -4.9
    for w, opt in ((w_sgd, opt_sgd), (w_adamw, opt_adamw)):
        w.grad = torch.tensor([g])
        opt.step()
        opt.zero_grad()
    if step <= 6 or step % 10 == 0:
        print(f"{step:>5} {g:>+7.1f} {w_sgd.item():>+9.4f} {w_adamw.item():>+9.4f}")
print("""
 -> SGD obeys every batch: +grad pushes it down, -grad pushes it back up.
    It zigzags around 0 forever, hostage to the noise.
 -> AdamW's momentum (beta1=0.9) is a running average of the last ~10 grads.
    The +5/-4.9 flip-flops cancel out, the tiny +0.05 signal survives, and
    the weight drifts steadily NEGATIVE (downhill on the true average).
    That's what "momentum smooths noisy mini-batch gradients" means.""")

# ================================================================ PART 3
print("=" * 74)
print("PART 3 — the W in AdamW: weight decay fires even when the gradient is ZERO")
print("=" * 74)

w_sgd = torch.tensor([5.0], requires_grad=True)
w_adamw = torch.tensor([5.0], requires_grad=True)
opt_sgd = torch.optim.SGD([w_sgd], lr=0.1)
opt_adamw = torch.optim.AdamW([w_adamw], lr=0.1, weight_decay=0.1)  # decay ON now

for step in range(50):
    for w, opt in ((w_sgd, opt_sgd), (w_adamw, opt_adamw)):
        w.grad = torch.tensor([0.0])      # loss doesn't care about this weight at all
        opt.step()
        opt.zero_grad()

print(f"start: w = 5.0, grad = 0.0 every step, 50 steps")
print(f"SGD   after 50 steps: {w_sgd.item():.4f}   (grad=0 -> -lr*0 = never moves)")
print(f"AdamW after 50 steps: {w_adamw.item():.4f}   (w *= (1 - lr*decay) each step)")
print("""
 -> SGD only listens to the gradient; grad=0 means frozen forever.
 -> AdamW additionally multiplies w by (1 - lr*weight_decay) every step:
    5.0 * 0.99^50 = 3.03. Unused weights don't sit at 5.0 — they shrink toward 0.
    "Decoupled" = this shrink is applied straight to w, NOT mixed into the
    gradient (that older trick, L2-in-the-grad, is plain Adam; the fix is the W).""")

# ================================================================ PART 4
print("=" * 74)
print("PART 4 — no magic: hand-write AdamW's first step, match torch exactly")
print("=" * 74)

w0, g = 3.0, 7.0
beta1, beta2, eps, decay = 0.9, 0.999, 1e-8, 0.01

# the real thing
w = torch.tensor([w0], requires_grad=True)
w.grad = torch.tensor([g])
torch.optim.AdamW([w], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=decay).step()

# the same math by hand (t=1) — mirrors minimind/train_pretrain_fp.py's adamw_update
m = (1 - beta1) * g                    # momentum:  0.9*0 + 0.1*grad
v = (1 - beta2) * g * g                # variance:  0.999*0 + 0.001*grad^2
m_hat = m / (1 - beta1 ** 1)           # bias correction (t=1)
v_hat = v / (1 - beta2 ** 1)
w_hand = w0 * (1 - lr * decay)         # 1) decoupled decay shrink
w_hand = w_hand - lr * m_hat / (v_hat ** 0.5 + eps)  # 2) normalized step

print(f"torch AdamW.step(): {w.item():.6f}")
print(f"by hand           : {w_hand:.6f}")
print(f"identical?          {abs(w.item() - w_hand) < 1e-6}")
print("""
 -> m_hat/sqrt(v_hat) is grad/|grad| = sign(grad) on the first step: that's WHY
    part 1 showed every AdamW step ~= lr regardless of gradient size.
    SGD:   w -= lr * grad                     (raw gradient, one global ruler)
    AdamW: w = w*(1-lr*decay) - lr * m_hat/(sqrt(v_hat)+eps)
           (smoothed gradient, per-weight ruler, plus a shrink toward 0)""")
