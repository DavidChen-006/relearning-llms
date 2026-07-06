# Why Infinitesimals Had to Exist — My Theory on the Origin of dy/dx

*David's observation, written up 2026-07-06, while tracing the heritage of the chain rule
(Leibniz → Linnainmaa → Werbos → Rumelhart/Hinton/Williams).*

## The theory

The naive way to measure a rate of change needs no calculus at all: pick a constant axis
(time), measure how far the thing traveled, and divide — distance over time. If the world
only contained constant-speed motion, this would be the end of the story: every interval
would report the same rate, and no finer tool would ever be needed.

But real motion isn't constant. A thrown object slows, and eventually stops. The moment you
admit that, division-over-an-interval breaks as a description of any *particular instant*:
the average over the whole flight is a fiction that matches no single moment of it. And the
changing itself has no smallest step — between the instant the object stops and the instant
just before, there is a finer difference in rate, and a finer one before that, without end.

So the rate of change must be defined *at an instant*, and the only path to an instant is
through ever-smaller slices of the interval. That is what forced the infinitely small
increment into existence: dy/dx is the tool that slices a trajectory arbitrarily finely —
a God's view of motion, able to freeze the flight at any frame and read off the rate that
lives only there.

Side observation: one *constant axis* (time, in the kinematic case) is what makes rate
measurable at all — that's the x of dy/dx. It generalizes beyond time to any variable you
slice along, which is exactly the step from Newton's motion-bound fluxions to Leibniz's
general-purpose notation.

## Historical verdict (was this the actual motivation?)

The core mechanism — *rates that vary defeat averages, therefore instants, therefore
infinitesimals* — is the true engine of calculus's birth. But it maps to the two inventors
differently:

- **Newton**: yes, almost exactly this path. Physicist chasing planetary motion and falling
  bodies; his "fluxions" were explicitly quantities flowing in time; instantaneous velocity
  was his home problem. (The ancient root is Zeno's arrow — what does motion even mean at
  an instant?)
- **Leibniz**: same underlying issue, different costume. His documented entry points:
  1. **The tangent problem** — given an arbitrary curve, find the line that just touches it
     at a point. A curve's steepness varies point to point, so no single ratio describes it.
  2. **Sums and differences of sequences** — differencing and summing undo each other;
     calculus as the continuous version (d = infinitely fine difference, ∫ = infinitely
     fine sum, inverse operations).
  3. **Philosophy of continuity** — "natura non facit saltus"; the continuum is infinitely
     divisible, so infinitesimals fit his picture of reality.

## The unification (follow-up observation)

**The tangent problem is the geometric version of Newton's kinematic problem.** They are the
same problem in two costumes:

- Newton: a *trajectory over time* whose rate varies — what is the speed at one instant?
- Leibniz: a *curve on paper* whose slope varies — what is the steepness at one point?

Plot Newton's motion as distance-vs-time and his question literally *becomes* the tangent
problem: instantaneous speed = slope of the tangent to the distance curve at that point.
One problem — "a varying rate has no honest average" — expressed once in physics and once
in geometry. Which explains why two people could invent the same calculus independently:
they were standing on two faces of the same mountain.
