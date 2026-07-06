# Euler: The Machine vs. the Output — and the Per-Weight Revelation

*David's session notes, 2026-07-07. First note in the Euler series (heritage stop between
Leibniz and Cauchy: 1670s → 1748 → 1847). This note resolved the "loss is a scalar but
rates are about change — they're incompatible" conflict.*

## 1. How Euler was thinking

Before Euler (1700s), math was about numbers and shapes — the answer 12, the circle on the
page. You calculated things and got results.

Euler said: forget the answers — the interesting object is **the machine that makes the
answers**. A vending machine: press B4, get chips; press C2, get soda. The machine itself —
the rule connecting every button to its snack — is a thing you can study, name, and do math
on, SEPARATELY from any one snack that falls out. He wrote the machine as **f(x)**: f is
the machine, x is which button you pressed.

This reorganized all of math: calculus stopped being "the study of curves" and became "the
study of machines" — how a machine's output responds when you turn its input knob.

The key sentence of Euler-thinking: **the machine and its output are two different things.**
The number that fell out has no personality — frozen, done. The machine has all the
personality: it's the thing with behavior, with "what would happen IF I pressed a different
button."

## 2. Applied to my situation

I was using one word — "loss" — for two different objects:

- **The output:** 8.76. One frozen snapshot. A frozen number can't have a rate — correct!
- **The machine:** L(w) — the rule taking ANY setting of the weights to a loss. THIS is
  the thing with a rate: "turn this knob up a hair, output drops 16 hairs" is a fact about
  the machine's behavior near the current knob-setting, not about the number 8.76.

The chain rule was never talking about the scalar. Forward pass = one button-press
(produces the snack). The tape it records describes the machine's local wiring — which is
why one run is enough to know how loss WOULD have differed, without rerunning.

One-liner: **loss the number = the machine's output today; dL/dw = the machine's
sensitivity to its knobs; rates belong to machines, never to outputs.**

Correction to keep: L(w) and dL/dw are TWO different machines —
L(w): knobs in → loss out (the landscape).
dL/dw: knobs in → tilt at that spot (the slope-reporter, manufactured from #1 the way
2x was manufactured from x²). Backward evaluates machine #2 at the current setting.

## 3. THE BIG REALIZATION: w is ONE weight, not the whole matrix

I had been reading dL/dw with W = the whole entire matrix. Wrong reading. **w is each
individual weight** — one single number inside a matrix. dL/dw asks, for THAT one number:

    "if I change this one weight by a hair, how much does the loss change by?"

Once w means one weight, blame makes complete sense:

- every individual weight gets its own answer → 1.9M answers, one per knob
- **blame = the per-weight importance table**: the weights with the biggest answers
  (big magnitude, e.g. very big negatives) are the ones whose individual change moves
  the loss the most — the ones to "blame"
- the shadow matrix is just this table filed in the same shape as the weight matrix,
  so each answer sits next to the number it describes

## 4. The clean stack (descent quarantined)

    Euler:    loss is a machine, not a number
    Leibniz:  chained machines' rates multiply → the output's rate can be decomposed
              through the internals
    blame:    that decomposition, evaluated at the current setting
              = the per-knob importance table (magnitude = importance, sign = direction)
    ─────────────── fence ───────────────
    Cauchy:   what to DO with the table — separate chapter, not yet
