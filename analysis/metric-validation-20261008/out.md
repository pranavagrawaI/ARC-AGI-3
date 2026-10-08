## Run-level metrics (full horizon)

| run | games | public_score | levels | levels_per_game | hazard | stuck_share | orient_frac | human_act_per_Mtok | human_act_per_game_hour | levels_per_Mtok | tok_per_game_s | med_act_over_base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| duck36-sub | 25 | 1.9 | 17 | 0.68 | 0.34 | 0.83 | 0.0628 | 303 | 8.64 | 11.9 | 7.92 | 0.966 |
| q38-commit | 25 | 3.29 | 22 | 0.88 | 0.44 | 0.748 | 0.132 | 305 | 11.7 | 11.5 | 10.7 | 0.801 |
| kf2 | 25 | 10.2 | 44 | 1.76 | 0.799 | 0.51 | 0.128 | 910 | 31.6 | 23 | 9.64 | 0.862 |
| c1 | 25 | 9.75 | 42 | 1.68 | 0.762 | 0.602 | 0.134 | 801 | 27 | 22.6 | 9.37 | 0.82 |
| p1 | 25 | 11.7 | 51 | 2.04 | 0.926 | 0.598 | 0.112 | 672 | 35.8 | 17.4 | 14.8 | 0.917 |
| p2 | 25 | 16.1 | 58 | 2.32 | 1.08 | 0.477 | 0.131 | 748 | 41.2 | 19.6 | 15.3 | 0.799 |
| r1 | 25 | 8.21 | 41 | 1.64 | 0.745 | 0.52 | 0.134 | 865 | 28.7 | 22.5 | 9.21 | 0.887 |
| g1 | 25 | 9.33 | 41 | 1.64 | 0.744 | 0.625 | 0.133 | 781 | 26.3 | 22 | 9.37 | 0.679 |
| g2 | 25 | 8.88 | 40 | 1.6 | 0.726 | 0.599 | 0.173 | 856 | 28.7 | 21.7 | 9.31 | 0.688 |
| dfz-control | 25 | 52.5 | 107 | 4.28 | 2.42 | 0.363 | 0.171 | 1.62e+03 | 159 | 24.7 | 27.3 | 0.595 |
| dfz-r2 | 25 | 47.4 | 100 | 4 | 2.63 | 0.371 | 0.195 | 1.74e+03 | 168 | 27.3 | 26.8 | 0.67 |
| dfz-s1 | 25 | 39.5 | 98 | 3.92 | 2.1 | 0.446 | 0.171 | 1.3e+03 | 121 | 22.4 | 26 | 0.654 |
| lean-cap4 | 10 | 11.1 | 18 | 1.8 | 1.8 | 0.475 | 0.0855 | 346 | 72.6 | 8.59 | 58.2 | 0.77 |
| lean-prune | 10 | 10.4 | 18 | 1.8 | 1.96 | 0.425 | 0.159 | 360 | 73.8 | 9.53 | 57 | 0.817 |

## T1: public-run metric vs hidden score (5 systems)

Order by hidden: duck36-sub 0.67, q38-commit 1.47, kf2 2.96, p1 3.08, dfz-control 23.87

| metric | spearman | log-log slope | residual sd (log) | KF->P1 predicted / actual hidden ratio | leave-one-out miss (duck36-sub, q38-commit, kf2, p1, dfz-control) |
|---|---|---|---|---|---|
| public_score | 1.00 | 1.02 | 0.28 | 1.16 / 1.04 | 1.05x 1.45x 1.28x 1.47x 2.20x |
| levels | 1.00 | 1.77 | 0.38 | 1.30 / 1.04 | 1.02x 1.65x 1.29x 1.74x 2.67x |
| levels_per_game | 1.00 | 1.77 | 0.38 | 1.30 / 1.04 | 1.02x 1.65x 1.29x 1.74x 2.67x |
| hazard | 1.00 | 1.72 | 0.24 | 1.29 / 1.04 | 1.15x 1.50x 1.07x 1.41x 1.71x |
| stuck_share | -0.90 | -3.92 | 0.38 | 0.53 / 1.04 | 1.22x 1.51x 2.02x 1.19x 2.30x |
| orient_frac | 0.60 | 2.94 | 0.87 | 0.68 / 1.04 | 10.17x 4.15x 1.45x 1.17x 6.12x |
| human_act_per_Mtok | 0.90 | 1.71 | 0.57 | 0.60 / 1.04 | 1.58x 2.53x 2.41x 1.11x 3.60x |
| human_act_per_game_hour | 1.00 | 1.14 | 0.25 | 1.15 / 1.04 | 1.24x 1.64x 1.15x 1.31x 1.70x |
| levels_per_Mtok | 0.80 | 3.09 | 0.85 | 0.42 / 1.04 | 2.06x 2.65x 4.81x 1.06x 5.96x |
| tok_per_game_s | 0.90 | 2.59 | 0.51 | 3.03 / 1.04 | 1.57x 1.36x 2.72x 1.56x 1.95x |
| med_act_over_base | -0.60 | -6.15 | 0.72 | 0.68 / 1.04 | 2.08x 2.75x 1.55x 2.93x 4.01x |

## T2: noise floor between byte-identical reruns (paired per level, solved in both)

| pair | k | n | geo ratio | t | sd(log) |
|---|---|---|---|---|---|
| dfz-control vs dfz-r2 | tok | 88 | 0.96 | -0.56 | 0.66 |
| dfz-control vs dfz-r2 | act | 88 | 1.02 | 0.52 | 0.38 |
| dfz-control vs dfz-r2 | sec | 88 | 0.93 | -0.73 | 0.90 |
| dfz-control vs dfz-r2 | orient | 86 | 1.07 | 1.06 | 0.58 |
| kf2 vs c1 | tok | 32 | 1.01 | 0.17 | 0.47 |
| kf2 vs c1 | act | 32 | 0.93 | -0.89 | 0.49 |
| kf2 vs c1 | sec | 32 | 1.14 | 1.37 | 0.55 |
| kf2 vs c1 | orient | 32 | 0.90 | -0.94 | 0.66 |

## T3: known differences, paired per level (B/A), common horizon

| A | B | horizon | known | levels A->B (shared games) | hazard B/A | tok/level geo (t) | act/level geo (t) | n |
|---|---|---|---|---|---|---|---|---|
| kf2 | dfz-control | full | hidden 8.1x | 44->107 | 3.03 | 0.83 (-2.3) | 0.69 (-3.4) | 44 |
| duck36-sub | q38-commit | full | hidden 2.2x | 17->22 | 1.29 | 1.37 (1.2) | 0.82 (-0.8) | 12 |
| kf2 | p1 | full | hidden 1.04x | 44->51 | 1.16 | 1.13 (1.2) | 0.99 (-0.1) | 38 |
| kf2 | p2 | full | same system as p1 | 44->58 | 1.35 | 1.18 (1.7) | 1.01 (0.1) | 39 |
| kf2 | g1 | full | null graft | 44->41 | 0.93 | 0.89 (-1.1) | 0.75 (-2.3) | 37 |
| kf2 | r1 | full | null repair | 44->41 | 0.93 | 1.04 (0.4) | 0.97 (-0.3) | 35 |
| dfz-control | dfz-s1 | full | prompt change, inconclusive | 107->98 | 0.87 | 1.07 (0.8) | 1.04 (0.9) | 83 |
| dfz-control | lean-cap4 | 3600 | Tycho lean arm, 10 games | 27->17 | 0.63 | 2.82 (6.4) | 0.84 (-1.7) | 16 |
| dfz-control | lean-prune | 3600 | Tycho lean arm, 10 games | 27->17 | 0.68 | 3.79 (7.7) | 0.95 (-0.5) | 16 |

## T4: early read. Run totals at 30/60 min vs final, over runs with a >=2 h horizon

- 30 min: spearman 0.54 over 8 runs; early share of final levels median 0.28
- 60 min: spearman 0.85 over 8 runs; early share of final levels median 0.58
