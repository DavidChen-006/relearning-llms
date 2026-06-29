"""demo_silu.py — graph SiLU(x) = x * sigmoid(x) right in the terminal (ASCII)."""
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def silu(x):
    return x * sigmoid(x)


# sample x from -6 to 6
xs = [-6 + i * 0.25 for i in range(49)]
ys = [silu(x) for x in xs]

W = len(xs)
H = 20
ymin, ymax = min(ys + [-1.0]), max(ys + [1.0])


def row_of(y):
    return int(round((ymax - y) / (ymax - ymin) * (H - 1)))


grid = [[" "] * W for _ in range(H)]

# x-axis (y=0) and y-axis (x=0)
zero_row = row_of(0.0)
for c in range(W):
    grid[zero_row][c] = "-"
x0 = min(range(W), key=lambda i: abs(xs[i]))
for r in range(H):
    grid[r][x0] = "|"
grid[zero_row][x0] = "+"

# the curve
for c, y in enumerate(ys):
    grid[row_of(y)][c] = "*"

print("SiLU(x) = x * sigmoid(x)      x: -6 .. +6\n")
for r in range(H):
    print("".join(grid[r]))

print("\nsample values:")
for x in [-6, -2, -1, 0, 1, 2, 6]:
    print(f"  SiLU({x:>2}) = {silu(x):6.3f}")
