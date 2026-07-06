# The Chain Rule's Purpose — Settled by A/B Test

*David's session notes, 2026-07-07. Fourth note in the Leibniz series. Method shift: chasing
PURPOSE, not mechanism — "give me something hard without it, then give it the chain rule."*

## 1. What "chain" means (settled)

Dependence in series: z depends on y, y depends on x — so z depends on x THROUGH y.
Function feeding function: z = f(y), y = g(x). The fraction picture (dy on top of one
ratio, under the next, relaying down the line) is the notation's portrait of "output of
one stage is the input of the next."

## 2. Bottom-up vs top-down (my confusion, resolved)

The equation dz/dx = dz/dy · dy/dx is an IDENTITY — true in both directions. The direction
that MATTERS depends on what you already know:

- top-down (know the composite's slope, factor it): permitted, but useless — if you
  already have dz/dx, you're done.
- **bottom-up (know each link's easy slope, multiply to build the whole): THE famous
  direction. What the chain rule is FOR.**

Analogy that kills the direction question: 6 = 2 × 3. Neither "top-down" nor "bottom-up" —
just true. You use it in the direction where you're poor and it makes you rich.

Key fact dissolving the "is dz/dx findable?" flip-flop: the composite's slope always EXISTS
(the composite is a real function), but the direct route may be unwritable or miserable —
the exists-vs-writable gap from exchange-rates-and-middlemen.md. The chain rule manufactures
the answer from the pieces regardless.

## 3. THE A/B TEST

Function: z = (x² + 3x)⁵

**Route A — no chain rule (expand first, then power rule):**

    (x² + 3x)⁵ = x¹⁰ + 15x⁹ + 90x⁸ + 270x⁷ + 405x⁶ + 243x⁵
    dz/dx     = 10x⁹ + 135x⁸ + 720x⁷ + 1890x⁶ + 2430x⁵ + 1215x⁴

Binomial grind, twelve coefficient computations, high blunder risk. Scaling sabotage:
50th power → 51 terms; add "+1" inside → dozens of cross-terms. Route A dies by
combinatorics.

**Route B — chain rule:**

    y = x² + 3x     dy/dx = 2x + 3      (trivial)
    z = y⁵          dz/dy = 5y⁴         (trivial)
    dz/dx = 5(x² + 3x)⁴ · (2x + 3)

Ten seconds. 50th power? Same two lines, one exponent changed.

**The referee** (numerical, at x = 2):

    Route A: 350,000    Route B: 350,000    measured wiggle: 350,000

Same slope, wildly different cost. The chain rule doesn't change what the slope IS;
it changes what it COSTS.

## 4. The knockout round: where Route A is IMPOSSIBLE

    z = e^(x²)      Route A: expand e^(x²) into a polynomial? It isn't one. No finite
                    expansion exists. Route A cannot even start.
    Route B:        y = x² (dy/dx = 2x),  z = e^y (dz/dy = e^y)
                    dz/dx = e^(x²) · 2x   — two lines, done.

Same for sin(x² + 3x), √(x² + 1), log(...): the moment one function sits INSIDE another of
a different family, expansion is off the table and the chain rule is the ONLY road.

## 5. Purpose, earned

1. **Compression:** hard slopes = products of easy slopes. Every individual link has a
   one-line derivative; composition is what makes functions hard; the chain rule un-makes it.
2. **Coverage:** extends differentiation from "functions I can expand" to every function
   built by composing simple ones — which is essentially all functions anyone writes,
   since writing a formula IS composing simple pieces.

Historical payload of (2): with the chain rule + product rule + a small table of
primitives, differentiation becomes MECHANICAL — any formula, however nested,
differentiable by recipe, no ingenuity required. A century of physics ran on that
mechanization.
