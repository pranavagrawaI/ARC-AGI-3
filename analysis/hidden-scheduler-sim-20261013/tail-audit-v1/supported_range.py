"""Does the prior scheduler gain need tails beyond the checked token range?

Supplementary CPU sensitivity after finite-range tail calibration. Replays the
same latent draws and hold tapes as audit-v1; only extrapolated (never empirical)
costs are limited. Conservative caps are not claims that longer solves cannot occur.
"""
import copy
import hashlib
import json
import math
import random

import calibrate_tail as t

h = t.h
HERE = t.HERE
CAPS = (None, 160_000, 200_000)


def main():
    target = HERE / "SUPPORTED-RANGE.json"
    manifest_path = HERE / "SUPPORTED-RANGE-MANIFEST.json"
    assert not target.exists() and not manifest_path.exists()
    manifest = {"reps": 100, "cohort": "c14_only", "cap_extrapolated_total_level_tokens": CAPS,
                "clock_and_rates": "identical to audit-v1", "policy": "unchanged out-of-fold HazardIndex",
                "seed_tapes": "identical to audit-v1 mixed primary",
                "interpretation": "Sensitivity bounds, not established absorbing barriers; observed costs retained at any value",
                "code_sha256": hashlib.sha256(t.Path(__file__).read_bytes()).hexdigest(),
                "source_manifest_sha256": hashlib.sha256((h.HERE / "MANIFEST.json").read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    data = json.loads((h.PARENT / "attempts.json").read_text())
    folds = h.folds_for(next(iter(data["games"].values())).keys())
    fits = {f: h.fit_fold(data, folds, f, "c14_only") for f in range(5)}
    worlds = {f: h.TaggedWorld(v["world"], v["train"], tail_scale=1.0, cost_scale=1.0) for f, v in fits.items()}
    game_fold = {g: f for f, games in enumerate(folds) for g in games}
    games = sorted(game_fold)
    rows = {str(cap): [] for cap in CAPS}
    extrapolated_costs = []

    def hazard(g, ctx):
        return fits[game_fold[g.inst.gid]]["hazard"].policy(g, ctx)

    class Holds:
        def draw(self, gid, hold_num, instance_num, seed):
            return fits[game_fold[gid]]["holds"].draw(gid, hold_num, instance_num, seed)

    for rep in range(100):
        rng = random.Random(f"audit-v1:mixed:{rep}")
        original = []
        for _ in range(110):
            gid = rng.choice(games)
            original.append(worlds[game_fold[gid]].draw(gid, rng))
        for inst in original:
            extrapolated_costs.extend(c for c, source in zip(inst.cost, inst.sources) if source == "extrapolated")
        for cap in CAPS:
            insts = copy.deepcopy(original)
            removed = 0
            for inst in insts:
                for k, source in enumerate(inst.sources):
                    if source == "extrapolated" and cap is not None and inst.cost[k] > cap:
                        inst.cost[k], inst.acts[k], inst.sources[k] = math.inf, 0, "unresolved"
                        removed += 1
            seed = f"audit-v1:holds:mixed:{rep}"
            pair = {"rep": rep, "removed_extrapolated_levels": removed,
                    "shipped": h.simulate(insts, h.sim.make_dfranzen(), holds=Holds(), seed=seed),
                    "hazard": h.simulate(insts, hazard, holds=Holds(), seed=seed)}
            for name in ("shipped", "hazard"):
                assert math.isclose(pair[name]["tokens"], h.SLOTS * h.RATE * h.HORIZON, abs_tol=1e-4)
            rows[str(cap)].append(pair)
    summaries = {cap: h.summarize(rs) for cap, rs in rows.items()}
    previous = json.loads((h.HERE / "RESULT.json").read_text())
    baseline = next(c["summary"] for c in previous["comparisons"] if c["cohort"] == "c14_only" and c["scenario"] == "base" and c["fold"] == "mixed")
    assert math.isclose(summaries['None']['shipped_score'], baseline['shipped_score'], abs_tol=1e-10)
    assert math.isclose(summaries['None']['hazard_score'], baseline['hazard_score'], abs_tol=1e-10)
    out = {"manifest": manifest, "exact_prior_base_reproduction": True, "comparisons": summaries, "rows": rows,
           "all_latent_extrapolated_cost_quantiles": [float(t.np.percentile(extrapolated_costs, p)) for p in (10, 50, 90)],
           "all_latent_extrapolated_costs_over_200k_fraction": sum(c > 200_000 for c in extrapolated_costs) / len(extrapolated_costs)}
    target.write_text(json.dumps(out, indent=2) + "\n")
    for cap, s in summaries.items():
        print(f"cap={cap} score={s['shipped_score']:.3f}->{s['hazard_score']:.3f} gain={s['relative_mean_gain']:+.2%}", flush=True)


if __name__ == "__main__":
    main()
