# MTP depth: bounded offline from retry1, no sweep needed (2026-10-16)

`bound.py` uses the two valid retry1 launches on the RTX PRO 6000 (M3A, M1; both healthy, zero errors,
though the occupancy gate failed). Output in `bound.out`:

- Per running request, M3 is +12.5% over M1 in tok/s for +48.3% token yield per round, so each extra
  draft token costs about 19% of a base round (c1/c0 = 0.189, linear cost model).
- M4 vs M3: at most **+3.7%** decode (q4 = q3, best feasible split), **-3.0%** at constant
  conditional decay; breakeven q4 = 0.315 against q3 ≈ 0.34.
- M2 vs M3: **-1.1%** at constant decay, range -4.8%..+9.9%. The top of the range needs q2 ≈ q1 = 0.77
  and q3 ≈ 0.09, a collapse at position 3 that a decaying MTP drafter does not show.

At f ≈ 0.8 of the critical path in decode and levels-time elasticity about 0.35, even +3.7% decode is
about +1% levels, far below hidden-draw noise (sd ≈ 3 points on ~26). Caveats: two noisy points (window
CV 22% and 14%), occupancy normalised by dividing by running count, linear round cost. None of these
plausibly turns a single-digit decode effect into a 20% bet.

Decision: retain M=3. No static MTP qualification run.
