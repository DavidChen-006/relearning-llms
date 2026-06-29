"""demo_rms.py — what dividing by RMS does, and WHY root-mean-square (not sum/mean/max).

3 parts:
  A) what normalization DOES (across different tensors)
  B) why SQUARE   (why not divide by sum or mean)
  C) why MEAN + ROOT (not just sum of squares)
"""
import torch


def rms(x):
    return x.pow(2).mean().sqrt()


print("=" * 60)
print("PART A — different SIZES, same PATTERN -> RMSNorm makes them equal")
print("=" * 60)
for v in [torch.tensor([1., 2., 3.]), torch.tensor([10., 20., 30.]), torch.tensor([100., 200., 300.])]:
    n = v / rms(v)
    print(f"  {str(v.tolist()):<22} RMS={rms(v):7.2f}  ->  normalized {[round(x, 3) for x in n.tolist()]}")
print("  All three collapse to the SAME vector: the SIZE is removed, the DIRECTION kept.")
print("  (that's the whole job: put different-scale vectors on one common scale)\n")

print("=" * 60)
print("PART B — why SQUARE?  (why not divide by the sum or the mean)")
print("=" * 60)
v = torch.tensor([5., -5., 5., -5.])
print(f"  v = {v.tolist()}   (clearly NOT a zero vector)")
print(f"  sum  = {v.sum():.1f}  and  mean = {v.mean():.1f}   -> dividing by these = divide by ZERO. broken.")
print(f"  RMS  = {rms(v):.1f}   -> v/RMS = {[round(x, 1) for x in (v / rms(v)).tolist()]}   works fine.")
print("  WHY: squaring makes every term positive, so + and - can't cancel out. That's the point of SQUARE.\n")

print("=" * 60)
print("PART C — why MEAN (not raw sum of squares) and why ROOT?")
print("=" * 60)
short = torch.tensor([3., 4.])
long = torch.tensor([3., 4., 3., 4., 3., 4.])
print(f"  short {short.tolist()}: sum-of-squares = {short.pow(2).sum():.0f},  RMS = {rms(short):.2f}")
print(f"  long  {long.tolist()}: sum-of-squares = {long.pow(2).sum():.0f},  RMS = {rms(long):.2f}")
print("  MEAN: the long vector's sum-of-squares is bigger ONLY because it has more numbers,")
print("        but its RMS is the SAME -> averaging removes the 'length advantage'.")
print("  ROOT: undoes the squaring (squares blew the units up), so RMS is in the same units as x,")
print("        which is why x/RMS lands around 1 instead of some squared scale.")
