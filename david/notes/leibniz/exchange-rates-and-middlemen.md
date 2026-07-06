# Exchange Rates, Identity Middlemen, and Whether the Middle Ever Mattered

*David's session notes, 2026-07-07. Third note in the Leibniz series — after
[calculus-origins.md](calculus-origins.md) (why infinitesimals) and
[differentials-and-chain-rule.md](differentials-and-chain-rule.md) (the machinery).*

## 1. My discovery: the chain rule is "omnipotent" (the identity middleman)

Even when dz depends on dx directly — no middleman anywhere — the chain rule still applies.
Insert a do-nothing middleman: let y be a COPY of x (y = x, so dy/dx = 1). Then:

    dz/dx = dz/dy · dy/dx = dz/dy · 1 = dz/dx   ✓ collapses to itself

So the chain rule isn't a special-case trick for when middlemen happen to exist — it's the
GENERAL grammar of dependence, degrading gracefully at the edges:

- copy link (y = x):      slope 1  → passes through invisibly
- constant link (y = c):  slope 0  → kills its road's contribution entirely
- real link:              slope whatever → scales the flow

(The identity function is the neutral element of composition — the ×1 of chaining.)

## 2. The exchange-rate explanation of WHY rates multiply (keeper)

A derivative is an exchange rate between tiny movements:

    dy/dx = units of y-movement received per unit of x-movement spent

Chain three currencies — yen → euro → dollar:

    dollars per euro: 2   (dz/dy)
    euros per yen:    3   (dy/dx)
    dollars per yen:  2 × 3 = 6   — and it COULDN'T be anything else

Why multiplication is FORCED: convert 1 yen → hold 3 euros → convert those → 6 dollars.
The middle currency is received from stage one and spent into stage two — THE SAME 3 euros
both times. That is the substance behind Leibniz's fraction cancellation: the dy in dz/dy
and the dy in dy/dx are literally the same tiny step, produced by one stage and consumed by
the next. Cancellation is honest because the middleman hands over exactly what it received.

Fine print (curves): exchange rates that vary with position are handled by making the
amounts infinitesimal, so each conversion happens at one frozen rate — the rate AT the
current point. (Why every rate must be evaluated where the chain actually passes.)

Multivariable version: several currencies, several conversion routes — multiply along each
route, ADD across routes.

## 3. My philosophical claim: the middleman was never necessary

The currency example is perfect but reveals something: nothing forces you through the euro.
A direct dollars↔yen quote can exist. Generalized:

**If a direct formula z(x) exists, the middleman was ontologically unnecessary — the chain
was a fact about how we COMPUTED, not about the quantities themselves.** And composition
guarantees the direct thing always exists: z = f(g(x)) IS a function of x, whether or not
anyone writes it as one formula. Middlemen are scaffolding: needed for the derivation,
dispensable in the result. Finding the direct formula is the PROOF of dispensability.

## 4. Where the middleman fights back (two counterpoints that sharpen the claim)

1. **Exists ≠ writable.** Some composites have no closed form — the direct formula cannot
   be written down — yet the chain rule still computes the exact rate from the pieces.
   So the middleman can be eliminable in principle (the composite function exists) while
   indispensable in practice (the only computable route runs through it). The gap between
   "exists" and "can be written" is a permanent feature of mathematics.

2. **Description vs. causation.** Currencies: the euro leg was our bookkeeping — the world
   permits a direct quote. Gears: pedal → gear → wheel — remove the gear and the wheel
   stops; the middle is physically real. The chain rule treats both identically: the
   formula is agnostic about whether the world has joints. Which dependencies are real and
   which are our descriptions is a question math deliberately doesn't answer — and one
   Leibniz the philosopher (necessary vs. contingent truths) would have recognized as his.
