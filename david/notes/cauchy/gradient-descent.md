# Cauchy: Gradient Descent — What to Do With the Table

*David's session notes, 2026-07-06. First note in the Cauchy series (heritage stop after
Leibniz and Euler: 1670s → 1748 → 1847). This is the far side of the fence — what to DO
with the per-weight importance table (blame) that backprop produces.*

## The equation

    w  ←  w  −  lr · (∂L/∂w)

new weight = old weight − learning rate × its gradient. Applied to every weight
independently, each step. (Vector form: θ ← θ − lr·∇L(θ).) This one line, looped, is
Cauchy's whole method and the body of optimizer.step().

## 1. It's a TEMPLATE, run per-weight — not one equation

Big misconception I had: I read w and grad as single fixed numbers, concluded "nothing
varies except lr, so lr must be the per-weight knob." Backwards.

- **w**: different for each weight (each has its own current value)
- **grad (∂L/∂w)**: different for each weight — THIS is the source of independence.
  Backprop gave each weight its own signal.
- **lr**: the SAME for every weight — one shared scalar, the ONLY global constant.

Same lr = 0.1 for all weights, yet every weight moves a different amount and even a
different direction — because the gradient differs, not lr.

## 2. Nothing is "optimized" in a single step — it's iteration, not solving

I kept looking for a variable to "solve for," like x in x². There isn't one. A single step
optimizes nothing — w, grad, lr are all given; you just compute and move.

Gradient descent is NOT "solve for the minimum." It's: take one step downhill, then
reassess. The thing that changes is **w, across steps**. w_new becomes w, and then —
the piece I was missing — **the gradient is RECOMPUTED at the new w.** It's a formula
(e.g. 2(w−5)), so it gives a different number at every position. That constant
re-measurement is the whole engine.

Why not just solve ∂L/∂w = 0 directly (real "find the minimum")? For a neural net that's
millions of coupled nonlinear equations with no closed-form solution — you literally
cannot. So you walk instead. The "optimization" is the TRAJECTORY over thousands of
steps, not any single equation.

"New loss is worse — now what?" Two cases, neither is undo:
1. lr small enough → loss guaranteed not worse (Cauchy's sum-of-squares: ΔL ≈ −lr·Σ(gradᵢ²),
   a negative because squares can't be negative).
2. loss got worse → lr was too big, you overshot the valley. Fix = smaller lr, not undo.

## 3. The x² unlock — the "purpose" is the gradient, and it's not random

I thought it seemed random: "just change weights, change weights, change weights." But
gradient descent clearly goes DOWN the slope — so something gives it direction.

My own x² example contained the answer. "To maximize x², pump x up." But how do you know
UP not down? At x = −3, pumping up makes x² SMALLER. You know to go up because **the slope
points that way** — you've internalized it so it feels automatic. The slope IS the
direction-giver, even in my simple example.

Gradient descent does the exact same thing — reads the slope to pick direction — just for
a trillion weights you can't eyeball. **The gradient IS "the direction of the slope."**
That's what all the backprop work was for: computing the compass needle for every weight.

Proof it's not random: random changes WANDER (loss bounces 25→32→25→20...); gradient
changes AIM (loss drops straight 25→12→6→3→1.4...). The only difference between the two
loops is the gradient. Change randomly = no intent. Change along the gradient = intent =
optimizing.

## 4. The core intuition: independent slides, no global planner

Right scope: think per individual weight. Each weight is independent — it has no idea other
weights exist. All it cares about is its one signal (its gradient). It slides against that
signal. Do this for all weights independently → new loss → backprop → new gradients →
slide again → repeat.

**There is no master planner, no global strategy, no variable-to-control.** The intelligence
isn't in any weight or in a plan — it EMERGES from a trillion dumb independent slides, each
guided by a gradient that backprop keeps recomputing. The "going down the 3D slope" purpose
is real but bottom-up: the sum of a trillion tiny local "which way does my ground tilt?"
decisions.

(Refinement: the weights aren't truly independent — they're coupled THROUGH the loss
(change one, everyone's gradient shifts next batch). But within a single step they act
independently; the coupling is handled automatically by the next backprop recomputing all
gradients fresh. So: independent slides within a step, re-coupled by the next backprop.)

## 5. Why lr can be universal, and why the minus sign is "push against"

- lr is one shared POSITIVE scalar. You never need it per-weight or negative.
- Each weight's own gradient (its SIGN and MAGNITUDE) decides which way and how far THAT
  weight goes. Sign = direction, magnitude = amount.
- The **minus sign** in the equation IS "push against," automatically, per weight:
    - gradient positive (loss rises if w↑) → w − lr·(+) → w goes DOWN ✓
    - gradient negative (loss rises if w↓) → w − lr·(−) → w goes UP ✓
  The fixed minus flips whatever sign the gradient has. lr stays positive and universal;
  the equation's structure handles direction for free.
- Flip the minus to a plus and you get **gradient ASCENT** — climb the function instead of
  descending it. Same equation, opposite sign, used when you want to maximize.

## The stack, now complete

    Euler:    loss is a machine, not a number
    Leibniz:  chained rates multiply → each weight's slope (∂L/∂w) is computable
    backprop: the affordable backward sweep that fills every weight's .grad (the compass)
    Cauchy:   each weight slides against its own signal — w ← w − lr·∂L/∂w — repeat

From "I don't know what blame even does" to this. The whole engine.
