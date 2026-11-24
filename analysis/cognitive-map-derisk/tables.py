"""Print markdown tables of key numbers from results/*.json (used to write summary.md)."""
import json
import os

import numpy as np

R = "results"
J = lambda n: json.load(open(os.path.join(R, n)))


def bl(v):
    i = int(np.nanargmax(v)); return f"{v[i]:.3f} (L{i - 1})"


def exp0():
    d = J("exp0_bf16_vs_fp32.json"); c = d["comparison"]
    print("### Exp 0\n| quantity | bf16 | fp32 |\n|---|---|---|")
    for k in ["peak_vram_gib", "kl_full_mean_t1000_2000", "kl_restr_mean_t1000_2000", "off_alpha_mass_mean_t1000_2000", "peak_probe_r2_ols", "peak_probe_r2_ridge1"]:
        print(f"| {k} | {d['bf16'][k]:.4f} | {d['fp32'][k]:.4f} |")
    print({k: (round(v, 5) if isinstance(v, float) else None) for k, v in c.items() if not isinstance(v, list)})


def exp1():
    d = J("exp1_convergence.json")
    print("### Exp 1 late window t in [3000,4000)\n| HMM | KL full | KL restricted | off-alphabet mass | 0-HMM | 1-HMM | 2 | 3 | 4 | 6 | 8 | 16 | 64 | k_eff (restricted) |\n|---|" + "---|" * 13)
    for n in ["Mess3", "Wing", "Arch"]:
        r = d[n]; L = r["late_t3000_4000"]; K = r["khmm_late_t3000_4000"]
        print(f"| {n} | {L['kl_full']:.3f} | {L['kl_restr']:.3f} | {L['off_alpha_mass']:.3f} | " + " | ".join(f"{K[k]:.4f}" for k in ["0", "1", "2", "3", "4", "6", "8", "16", "64"]) + f" | {r['k_eff_restr']} |")
    print("\n| HMM | metric | " + " | ".join(d["Mess3"]["binned"]["kl_restr"]) + " |\n|---|---|" + "---|" * 6)
    for n in ["Mess3", "Wing", "Arch"]:
        for m in ["kl_full", "kl_restr", "off_alpha_mass"]:
            print(f"| {n} | {m} | " + " | ".join(f"{v['mean']:.3f}±{v['sem']:.3f}" for v in d[n]["binned"][m].values()) + " |")


def exp2():
    d = J("exp2_probes_controls.json")
    print("### Exp 2 belief probe, best layer (ridge, lambda by inner validation) / OLS min-norm best layer")
    print("| HMM | target | random 400/1600 | random 1000/1000 | contiguous | cross-sequence |\n|---|---|---|---|---|---|")
    for n in ["Mess3", "Wing", "Arch"]:
        ap = d[n]["acts_probe"]
        for t in ["belief", "eta_k5", "eta_k20", "shuffled_belief", "dirichlet_random", "cum_freq", "win_freq_50", "win_freq_200"]:
            print(f"| {n} | {t} | " + " | ".join(f"{bl(ap[p][t]['ridge_r2'])} / {max(ap[p][t]['ols_r2']):.3f}" for p in ["rand400", "rand1000", "contig", "xseq"]) + " |")
        for f, r in d[n]["feature_baselines"].items():
            print(f"| {n} | features: {f} | " + " | ".join(f"{r[p]['belief']['ridge_r2']:.3f}" for p in ["rand400", "rand1000", "contig", "xseq"]) + " |")
    print("\nlambda chosen (belief, cross-sequence) per layer:")
    for n in ["Mess3", "Wing", "Arch"]:
        ls = d[n]["acts_probe"]["xseq"]["belief"]["lam_sel"]
        print(n, [max(set(l), key=l.count) for l in ls])
    print("\n### 0-HMM / 1-HMM sequence controls (best layer ridge)")
    print("| HMM | sequences | split | belief | eta_k1 | eta_k2 | eta_k5 | eta_k10 | eta_k20 |\n|---|---|---|---|---|---|---|---|---|")
    for n in ["Wing", "Arch"]:
        c = d[n]["ctrl01"]
        for law in "01":
            for p in ["rand400", "contig"]:
                cc = c[f"{law}HMM_sequences"][p]
                print(f"| {n} | {law}-HMM | {p} | " + " | ".join(bl(cc[t]["ridge_r2"]) for t in ["belief", "eta_k1", "eta_k2", "eta_k5", "eta_k10", "eta_k20"]) + " |")
    print("\n### trajectory (best layer ridge)")
    print("| HMM | split | " + " | ".join(d["Wing"]["trajectory"]) + " |\n|---|---|" + "---|" * 5)
    for n in ["Mess3", "Wing", "Arch"]:
        tr = d[n]["trajectory"]
        for p in ["rand20", "contig", "xseq"]:
            print(f"| {n} | {p} | " + " | ".join(bl(tr[w][p]["belief"]["ridge_r2"]) for w in tr) + " |")


def exp3():
    d = J("exp3_cross_symbol_icl.json")
    if "transfer" in d:
        print("### Exp 3 transfer, per HMM best layer for each quantity")
        keys = ["A_to_A_heldout_seq_r2", "B_to_B_fresh_heldout_seq_r2", "WA_pooled_to_B_r2_identity", "WA_pooled_to_B_r2_bestperm",
                "WA_single_seq0_to_B_r2_identity", "affine_refit_B_from_A_subspace_r2", "affine_refit_B_from_B_subspace_r2",
                "affine_refit_B_from_random_subspace_r2", "procrustes_r2_WA_outputs_on_B", "cka_XA_QA_vs_XB_QB_matched", "cka_full_acts_A_vs_B_matched"]
        for n, r in d["transfer"].items():
            L = r["by_layer"]
            print(f"\n{n}")
            for k in keys:
                print(f"  {k}: {bl([x[k] for x in L])}")
            pc = [x["principal_cosines_WA_WB"] for x in L]; ph = [x["principal_cosines_WA_half1_half2"] for x in L]
            print("  max principal cosine W_A vs W_B by layer:", [round(max(p), 2) for p in pc])
            print("  max principal cosine W_A(seqs 0-1) vs W_A(seqs 2-4):", [round(max(p), 2) for p in ph])
            print("  random-subspace max cosine:", round(L[8]["principal_cosine_random_max_mean"], 3))
            print("  per-layer WA->B identity:", [round(x["WA_pooled_to_B_r2_identity"], 2) for x in L])
            print("  per-layer A->A:", [round(x["A_to_A_heldout_seq_r2"], 2) for x in L])
            print("  per-layer B->B:", [round(x["B_to_B_fresh_heldout_seq_r2"], 2) for x in L])
            print("  per-layer refit-from-A-subspace:", [round(x["affine_refit_B_from_A_subspace_r2"], 2) for x in L])
            b = r["behaviour"]
            for a in "AB":
                print(f"  behaviour {a}: KL_restr", {w: round(v["mean"], 3) for w, v in b[a]["kl_restr"].items()}, "off", {w: round(v["mean"], 3) for w, v in b[a]["off_alpha_mass"].items()})
    if "switch" in d:
        s = d["switch"]
        print("\n### Exp 3 switch (Wing)")
        for v in ["scratch", "nosep", "sep"]:
            r = s[v]; b = r["behaviour_B_segment"]
            print(v, "KL_restr", {w: round(x["mean"], 3) for w, x in b["kl_restr"].items()})
            print(v, "KL_full", {w: round(x["mean"], 3) for w, x in b["kl_full"].items()})
            print(v, "off-B mass", {w: round(x["mean"], 3) for w, x in b["off_alpha_mass"].items()}, "mass on A", {w: round(x["mean"], 3) for w, x in b["mass_other_alpha"].items()})
            print(v, "W_A fixed best layer", {w: bl([f[w] for f in r["WA_fixed_probe_r2_by_layer"]]) for w in r["WA_fixed_probe_r2_by_layer"][0]})
            print(v, "fresh W_B best layer", {w: bl([f[w] for f in r["fresh_WB_loo_seq_r2_by_layer_best_lambda"]]) for w in r["fresh_WB_loo_seq_r2_by_layer_best_lambda"][0]})


def exp4():
    d = J("exp4_finetune_reuse.json")
    for reg, t in d["train"].items():
        print(f"### train {reg} lr={t['lr']}")
        for st, c in t["checkpoints"].items():
            print(f"  step {st}: A KLfull {c['A_kl_full_late']:.4f} KLr {c['A_kl_restr_late']:.4f} | B KLfull {c['B_kl_full_late']:.3f} KLr {c['B_kl_restr_late']:.3f} massA {c['B_mass_other_alpha_late']:.3f} massB {1 - c['B_off_alpha_mass_late']:.3f}; early B massA {c['B_mass_other_alpha_early_0_250']:.3f} massB {1 - c['B_off_alpha_mass_early_0_250']:.3f}")
    for reg, data in sorted(d.get("ckpt_probe", {}).items()):
        print(f"exp4_ckpt_probe_{reg}", {s: (bl(r["A_heldout_seq_r2"]), bl(r["WA_to_B_late_2000_4000"]), bl(r["WA_to_B_early_0_500"])) for s, r in data.items()})
    for tag, e in d["eval"].items():
        print(f"\n### eval {tag}")
        for a in "AB":
            b = e[f"behaviour_{a}"]
            for m in ["kl_full", "kl_restr", "off_alpha_mass", "mass_other_alpha"]:
                print(f"  {a} {m}", {w: round(v["mean"], 3) for w, v in b[m].items()})
        L = e["by_layer"]
        print("  A held-out-seq R2 best", bl([x["A_heldout_seq_r2"] for x in L]))
        for k in ["WA_on_B", "WA_on_B_bestperm", "fresh_WB", "affine_refit_A_subspace"]:
            print(f"  {k}", {w: bl([x[k][w] for x in L]) for w in L[0][k]})


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        globals()[f]()
