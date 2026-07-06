# Working Through the Differential and the Chain Rule

*David's session notes, 2026-07-06. Companion to [calculus-origins.md](calculus-origins.md) —
that one is why infinitesimals had to exist; this one is how the machinery actually works,
re-derived from the definition up.*

## 1. The definition, restated plainly

dy/dx = take a very small step on x; how much does y move?

It's a ratio of two tiny steps. For kinematics this is obviously meaningful (tiny time step,
tiny distance step). The question was whether the *formulas* built on it really make sense.

## 2. Sanity-checking the formula: why d(x²)/dx = 2x makes sense

First reaction: "2x? does that even mean anything?" Then checking it against points:

- at x = 0 the parabola is flat → slope 0 = 2·0 ✓
- at x = 1 it's climbing → slope 2 = 2·1 ✓
- the further out, the steeper → slope keeps growing with x ✓

Realization: **the derivative of a curve must itself be a function of x, not a number** —
because a curve's steepness varies by location. 2x isn't a weird answer; it's the only
*kind* of answer a curve could have.

## 3. My theory ("they found the formulas empirically") — corrected

I guessed the power rule n·x^(n-1) was found by trial: a formula that happened to fit.
Wrong — it's **derived**, and the derivation is four lines of ordinary algebra:

```
y = x².  Step x by the infinitesimal dx, expand the square:

    y + dy = (x + dx)² = x² + 2x·dx + (dx)²

    so  dy = 2x·dx + (dx)²
```

Leibniz's move: (dx)² is an infinitely small piece OF an infinitely small thing —
negligible even by infinitesimal standards. Drop it:

```
    dy = 2x·dx      →      dy/dx = 2x
```

No experiments, no fitting. The 2x falls out of expanding a square and discarding the crumb.
Same pattern proves x³ → 3x² (expand, keep the single-dx term, drop crumbs), and n·x^(n-1)
in general.

**Connection:** this derivation IS the wiggle test in algebra form. The measured slopes
(-7, -7.9, -7.99, ... → -8) converge exactly because the (dx)²-sized error dies as the
step shrinks. Measurement and algebra agree because they are the same argument.

## 4. The chain rule's fine print: dependence must be routed THROUGH the middleman

My observation: for dz/dx = dz/dy · dy/dx to work, dz and dx must be dependent *through* dy —
otherwise the equation can't hold.

Confirmed, and it's the rule's condition of applicability, not a triviality: the product form
assumes z hears about x **only via y** — one road. If z also depends on x directly
(a second road bypassing y), the pure product undercounts and the rule grows a sum:

```
one road:    dz/dx = dz/dy · dy/dx
two roads:   dz/dx = dz/dy · dy/dx  +  (the direct road's contribution)
```

That sum-over-roads version is the multivariable chain rule. Handling MANY roads through
MANY variables efficiently is exactly the 1970 question (next stop on the heritage line:
Linnainmaa).

## The stack so far

definition (tiny-step ratio) → slopes are functions because curves bend → formulas come from
expand-and-drop-the-crumbs → chain rule applies when dependence is routed through the
middleman, and grows a sum when roads multiply.
