"""Bound static MTP depth M2/M4 against M3 from the two valid retry1 points (M3A, M1). No GPU.

Per-request throughput u = U / mean_R removes most of the occupancy gap (8.05 vs 8.41 running).
Round cost is modelled as c0 + c1*M (one target verify + M drafter steps), fitted to M1/M3:
  u3/u1 = (A3/A1) * cost1/cost3.  Acceptance q_i = P(first i drafts survive), q1 >= q2 >= q3 >= q4.
"""
import json, itertools
R = json.load(open("../mtp-depth-retry1-readout-20261008/recomputed-verdict.json"))["launches"]
m3, m1 = R["M3A"], R["M1"]
A3, A1 = m3["accept_len"], m1["accept_len"]
u3, u1 = m3["U"] / m3["mean_R"], m1["U"] / m1["mean_R"]
cost_ratio = (A3 / A1) / (u3 / u1)          # cost3/cost1
k = (cost_ratio - 1) / (3 - cost_ratio)      # c1/c0 from (1+3k)/(1+k) = cost_ratio
cost = lambda M: 1 + k * M
q1 = A1 - 1; rest = A3 - 1 - q1              # q2 + q3
print(f"u3={u3:.1f} u1={u1:.1f} tok/s/request (+{u3/u1-1:.1%}); token yield +{A3/A1-1:.1%}; cost3/cost1={cost_ratio:.3f}; c1/c0={k:.3f}")
print(f"q1={q1:.3f}  q2+q3={rest:.3f}")
rows = []
for q2 in [x / 1000 for x in range(int(rest / 2 * 1000), int(q1 * 1000) + 1)]:
    q3 = rest - q2
    if not (q1 >= q2 >= q3 >= 0): continue
    A2 = 1 + q1 + q2
    m2 = (A2 / A3) * cost(3) / cost(2) - 1
    # M4: q4 in [0, q3]; geometric continuation q4 = q3 * (q3/q2) is the 'typical' case
    m4_hi = ((A3 + q3) / A3) * cost(3) / cost(4) - 1
    m4_geo = ((A3 + q3 * q3 / q2) / A3) * cost(3) / cost(4) - 1
    rows.append((q2, q3, m2, m4_geo, m4_hi))
geo = min(rows, key=lambda r: abs(r[1] / r[0] - r[0] / q1))  # constant conditional decay q2/q1 = q3/q2
print(f"breakeven q4 for M4 = {A3*(cost(4)/cost(3)-1):.3f}")
print(f"M2 vs M3 over feasible (q2,q3): {min(r[2] for r in rows):+.1%} .. {max(r[2] for r in rows):+.1%}")
print(f"M4 vs M3 upper bound (q4=q3): {min(r[4] for r in rows):+.1%} .. {max(r[4] for r in rows):+.1%}")
print(f"constant-decay point q2={geo[0]:.3f} q3={geo[1]:.3f}: M2 {geo[2]:+.1%}, M4 {geo[3]:+.1%} (q4=q3 bound {geo[4]:+.1%})")
