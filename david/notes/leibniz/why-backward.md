# Why You Have to Go Backwards

*David's realization, 2026-07-07. Fifth note in the Leibniz series — and the bridge out of
it: this is the doorstep of Linnainmaa (1970).*

## The realization

You have many milestones of weights (attention's, the MLP's, lm_head's...). For each one,
the only question is: **when THIS weight changes, how does the loss move?** — dL/dw.

And the chain rule only traverses the dependency chain. dL/dw is the product of local rates
along the road FROM w TO the loss — w's downstream. Anything computed *before* w in the
forward pass is not on that road, so it appears nowhere in w's equation:

    dL/d(lm_head.weight)  involves the grading ops — and NOTHING else.
    Attention, RoPE, the embeddings: not in the chain, not in the product.

Visible in the algebra itself: if the product you need is dL/da · da/dw, then factors that
would sit further right (db/dc · dc/dx ...) simply are not in the equation. Not only
mathematically proven — it makes sense: things upstream of the weight don't carry its
influence to the loss.

## The corollary that forces the backward direction

Every weight's chain ENDS at the same place — the loss. So the 1.9M chains (one per
weight-number) are not 1.9M unrelated products: they SHARE their loss-side segments.

    chain for lm_head.weight :                    grading ops · (its local link)
    chain for norm gamma     :          grading ops · lm_head · (its local link)
    chain for q_proj         :  grading ops · lm_head · norm · ... · (its local link)
                                 └────────── shared prefixes ──────────┘

Start the walk at the SHARED end (the loss), compute each shared segment ONCE, and fan out
toward the weights — every weight's gradient falls out as the walk passes its driveway.
Start anywhere else and you recompute the shared segments millions of times.

That is why the sweep runs backward: not convention, not symmetry with forward — it is the
direction in which the chains share, and sharing is the only thing that makes 1.9M
(or 744B) chain-rule products affordable in one pass.

## Where this leads

Making this sharing precise — in what order do you multiply so every shared segment is
computed exactly once — is reverse-mode automatic differentiation: Linnainmaa, 1970.
Next stop on the heritage line.
