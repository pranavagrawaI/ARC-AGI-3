"""Discrete-event simulator of the dfranzen priority gate under public and hidden contention.

Model (all in generated tokens; wall time only through the per-slot token rate and the cap):
  * A game is a sequence of levels; level k of a simulated game costs cost[k] generated tokens to solve
    and is solved with acts[k] actions. Costs come from CostModel (trajectory resampling of logged
    public-25 runs, censored levels completed from the per-(game, level) Kaplan-Meier pool with a
    constant-hazard tail beyond the last observed solve).
  * `slots` games hold a slot at once. A holder generates a hold quantum (tokens between context trims,
    measured from the 14-stream request logs: first hold ~44K, later ~28K) and then hands the slot over,
    exactly where the shipped gate does (_maybe_handover after a trim, c14_notebook.py 6946-6990).
  * At every handover the waiting games are re-priced (ARC3_PRIORITY_REFRESH_QUEUE=1, _pump 3352-3365)
    and the highest priority wins; ties by arrival. Unstarted games sit in the 2,000,000 band minus
    queue position (4264, 8423) so every game starts before any trimmed game returns.
  * Per-slot effective rate (generation incl. tool/env/prefill gaps) r(n) = R14 * sqrt(14/n), fitted to
    the logged runs (53.2 tok/s at 14 streams, 62.9 at 10).
Scoring is the taaf port of the ARC-AGI-3 scorer (game.py 381-412).
"""
from __future__ import annotations

import heapq
import json
import math
import random
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UNSTARTED_BASE = 2_000_000  # _PRIORITY_UNTRIMMED_BASE, c14_notebook.py 3394


# ----------------------------------------------------------------------------------- scorer
def game_score(n_levels: int, base: list[int], levels_completed: int, actions_per_level: list[int]) -> float:
    """Port of taaf GameRun._compute_final_score (tufa-arc-agi-framework/src/taaf/game.py 381-412)."""
    if base is None or n_levels == 0:
        return 0.0
    total_score = 0.0
    total_weights = max_weights = 0
    for i in range(n_levels):
        w = i + 1
        total_weights += w
        acts = actions_per_level[i] if i < len(actions_per_level) else 0
        ls = min(115.0, (base[i] / acts) ** 2 * 100) if (i < levels_completed and acts > 0) else 0.0
        if ls > 0:
            max_weights += w
        total_score += ls * w
    if total_weights == 0:
        return 0.0
    return min(total_score / total_weights, max_weights / total_weights * 100)


# -------------------------------------------------------------------- shipped gate (port)
# priority_scheduler.py as patched into c14_notebook.py lines 262-404.
TAIL_LOOKUP_REMAINING = {n: (8.0,) * (n - 3) + (7.0, 5.0, 0.0) for n in range(6, 11)}  # 300-302


def parse_total_levels(n) -> int | None:  # 305-315
    if n is None:
        return None
    return min(10, max(6, int(n)))


def priority_value(level: int, actions: int, tokens: float, total_levels: int | None, tail_fraction: float | None,
                   *, token_scale: float = 62000.0, action_scale: float = 115.0, action_weight: float = 0.25,
                   human_actions: float = 25.0, normalize_score: bool = True, tail_lookup: str = "remaining") -> int:
    """priority_value() at 347-404 with the shipped env: SCORE_NORMALIZATION=1, TAIL_LOOKUP=remaining,
    TAIL_FADE=1 (tail_fraction supplied), PACE=0 (pace=1), endgame off (ARC3_ENDGAME_START_MINUTES unset).
    Defaults of the env getters: action_scale 115 (3412), token_scale 62000 (3416), action_weight 0.25 (3428),
    human_actions 25 (3462)."""
    level = max(1, level)
    total = parse_total_levels(total_levels) or 10
    actions, tokens = max(0, actions), max(0.0, tokens)
    pace = 1.0
    efficiency = (human_actions / (human_actions + actions)) ** 2
    current_value = level * efficiency
    if normalize_score:
        current_value *= 55.0 / (total * (total + 1) / 2)
    if tail_lookup == "remaining":
        tail = TAIL_LOOKUP_REMAINING[total][min(total, level) - 1]
    else:
        raise NotImplementedError
    if tail_fraction is not None:
        tail *= min(1.0, max(0.0, tail_fraction))
    chance = (action_weight * 0.5 ** ((actions / action_scale) ** 2)
              + (1.0 - action_weight) * 0.5 ** ((tokens / (token_scale * pace)) ** 2))
    value = (current_value + tail) * chance / pace * 100
    return max(1, int(value))


def tail_fraction_at(now: float, horizon: float, fraction: float = 0.2) -> float:
    """_snapshot_priority 3266-3282: window = fraction * (deadline - start); remaining / window."""
    window = fraction * horizon
    return max(0.0, horizon - now) / window if window > 0 else 0.0


# ---------------------------------------------------------------------------- cost model
def km_atoms(samples):
    """samples [(t, solved, actions)] -> (atoms [(t, mass, actions)], residual mass, last event t)."""
    s = sorted(samples, key=lambda x: x[0])
    n = len(s)
    surv, atoms, i, last = 1.0, [], 0, 0.0
    while i < n:
        t = s[i][0]
        group = []
        while i < n and s[i][0] == t:
            group.append(s[i])
            i += 1
        d = [g for g in group if g[1]]
        at_risk = n - (i - len(group))
        if d:
            m = surv * len(d) / at_risk
            for g in d:
                atoms.append((t, m / len(d), g[2]))
            surv -= m
            last = t
    return atoms, max(0.0, surv), last


@dataclass
class GameInst:
    gid: str
    n: int
    base: list
    cost: list
    acts: list


class CostModel:
    """Per-game level costs from logged runs.

    draw(): pick one logged realization (run) of the game; keep its solved levels' (tokens, actions);
    complete its censored level from the KM of all runs' attempts at that (game, level) conditional on
    exceeding the censor point; draw deeper levels from the unconditional (game, level) KM. KM residual
    mass is placed at last-event + Exp(tail hazard). Levels never reached in any run use the pooled
    attempts at the same level index from games in the same difficulty tercile.
    """

    def __init__(self, data: dict, *, runs: list[str] | None = None, tail_scale: float = 1.0,
                 cost_scale: float = 1.0, tail_start: float = 80_000.0, tail_mode: str = "exp",
                 run_scale: dict | None = None):
        # run_scale rescales a donor run's tokens to the target config (e.g. 10-stream/128K-context runs
        # spend ~17% fewer tokens per matched level than the shipped 14-stream/92K config).
        att = [dict(a, tokens=a["tokens"] * (run_scale or {}).get(a["run"], 1.0))
               for a in data["attempts"] if runs is None or a["run"] in runs]
        self.runs = sorted({a["run"] for a in att})
        self.meta = {}
        for r in self.runs:
            for g, m in data["games"][r].items():
                self.meta[g] = (m["n_levels"], m["base"])
        self.pool = defaultdict(list)
        self.real = defaultdict(dict)
        tok_by_game, act_by_game = defaultdict(float), defaultdict(float)
        for a in att:
            self.pool[(a["game"], a["level"])].append((a["tokens"], a["solved"], a["actions"]))
            self.real[a["game"]].setdefault(a["run"], []).append(a)
            tok_by_game[a["game"]] += a["tokens"]
            act_by_game[a["game"]] += a["actions"]
        self.apt = {g: act_by_game[g] / max(1.0, tok_by_game[g]) for g in tok_by_game}
        exposure = sum(t - tail_start for t, _, _ in (x for v in self.pool.values() for x in v) if t > tail_start)
        events = sum(1 for v in self.pool.values() for t, e, _ in v if t > tail_start and e)
        self.lam = tail_scale * events / exposure if exposure else 0.0
        self.tail_mode = tail_mode
        self.cost_scale = cost_scale
        # difficulty terciles by fraction of attempts solved
        frac = {}
        for g in self.real:
            v = [x for k in range(self.meta[g][0]) for x in self.pool.get((g, k), [])]
            frac[g] = sum(e for _, e, _ in v) / max(1, len(v))
        order = sorted(frac, key=frac.get)
        self.tercile = {g: i * 3 // len(order) for i, g in enumerate(order)}
        self.fallback = defaultdict(list)
        for (g, k), v in self.pool.items():
            self.fallback[(self.tercile[g], min(k, 5))] += v
        self.games = sorted(self.real)

    def _extrapolate(self, start: float, g: str, rng: random.Random):
        if self.lam <= 0 or self.tail_mode == "none":
            return math.inf, 0
        t = start + rng.expovariate(self.lam)
        return t, max(1, int(round(t * self.apt[g])))

    def _draw_level(self, g: str, k: int, censor: float, rng: random.Random):
        samples = self.pool.get((g, k)) or self.fallback[(self.tercile[g], min(k, 5))]
        samples = [s for s in samples if s[0] > censor]
        if not samples:
            return self._extrapolate(censor, g, rng)
        atoms, resid, last = km_atoms(samples)
        u = rng.random()
        acc = 0.0
        for t, m, a in atoms:
            acc += m
            if u < acc:
                return t, a
        return self._extrapolate(max(censor, last), g, rng)

    def draw(self, g: str, rng: random.Random, run: str | None = None) -> GameInst:
        n, base = self.meta[g]
        run = run or rng.choice(sorted(self.real[g]))
        rows = sorted(self.real[g][run], key=lambda a: a["level"])
        cost, acts = [], []
        censor = 0.0
        for a in rows:
            if a["solved"]:
                cost.append(a["tokens"])
                acts.append(a["actions"])
            else:
                censor = a["tokens"]
                break
        while len(cost) < n:
            c, a = self._draw_level(g, len(cost), censor, rng)
            censor = 0.0
            cost.append(c)
            acts.append(a)
        return GameInst(g, n, base, [c * self.cost_scale for c in cost], acts)


# ------------------------------------------------------------------------------ simulator
@dataclass
class GState:
    inst: GameInst
    idx: int
    k: int = 0              # current level (0-based); == n when won
    t_on: float = 0.0       # tokens on the current level
    tokens: float = 0.0     # total generated
    apl: list = field(default_factory=list)
    started: bool = False
    holds: int = 0
    solve_times: list = field(default_factory=list)
    arrival: int = 0

    @property
    def won(self) -> bool:
        return self.k >= self.inst.n

    @property
    def a_on(self) -> int:
        """Actions on the current level, linear in tokens toward the level's realized action count."""
        if self.won:
            return 0
        c, a = self.inst.cost[self.k], self.inst.acts[self.k]
        if not math.isfinite(c) or c <= 0:
            return int(self.t_on * 0.0012)
        return int(a * min(1.0, self.t_on / c))


@dataclass
class Ctx:
    now: float
    horizon: float
    rate: float


class Holds:
    def __init__(self, path: Path = HERE / "holds.json"):
        h = json.loads(path.read_text())
        self.first = sorted(v[0] for r in h.values() for v in r.values() if len(v) > 1)
        self.later = sorted(x for r in h.values() for v in r.values() for x in v[1:-1])

    def draw(self, first: bool, rng: random.Random) -> float:
        return rng.choice(self.first if first else self.later)


def rate_fn(n_active: int, r14: float) -> float:
    return min(90.0, r14 * math.sqrt(14.0 / max(1, n_active)))


def simulate(insts: list[GameInst], policy, *, slots: int, horizon: float, r14: float, holds: Holds,
             rng: random.Random, record: bool = False) -> dict:
    games = [GState(inst=x, idx=i, apl=[0] * x.n) for i, x in enumerate(insts)]
    waiting: list[GState] = list(games)
    seq = 0
    for g in games:
        g.arrival = seq
        seq += 1
    active: list = []  # heap (end, seq, game)
    free = slots
    now = 0.0

    def start_hold(g: GState, now: float):
        nonlocal seq
        n_active = slots - free + 1
        rate = rate_fn(n_active, r14)
        q = holds.draw(not g.started, rng)
        g.started = True
        g.holds += 1
        budget = min(q, max(0.0, horizon - now) * rate)
        used = 0.0
        while used < budget and not g.won:
            need = g.inst.cost[g.k] - g.t_on
            if need <= budget - used:
                used += need
                g.apl[g.k] = g.inst.acts[g.k]
                g.solve_times.append(now + used / rate)
                g.k += 1
                g.t_on = 0.0
            else:
                g.t_on += budget - used
                used = budget
        if not g.won:
            g.apl[g.k] = g.a_on
        g.tokens += used
        seq += 1
        heapq.heappush(active, (now + used / rate if used > 0 else now + 1e-6, seq, g))

    def pump(now: float):
        nonlocal free
        ctx = Ctx(now, horizon, rate_fn(slots, r14))
        while free > 0 and waiting:
            best, best_key = None, None
            for g in waiting:
                p = UNSTARTED_BASE - g.idx if not g.started else policy(g, ctx)
                key = (p, -g.arrival)
                if best_key is None or key > best_key:
                    best, best_key = g, key
            waiting.remove(best)
            start_hold(best, now)
            free -= 1

    pump(0.0)
    while active:
        end, _, g = heapq.heappop(active)
        now = end
        free += 1
        if now < horizon and not g.won:
            g.arrival = seq = seq + 1
            waiting.append(g)
        if now < horizon:
            pump(now)
    scores = [game_score(g.inst.n, g.inst.base, g.k, g.apl) for g in games]
    out = dict(levels=sum(g.k for g in games), mean_score=float(np.mean(scores)),
               tokens=[g.tokens for g in games], levels_by_game=[g.k for g in games], scores=scores,
               won=[g.won for g in games])
    if record:
        out["solve_times"] = sorted(t for g in games for t in g.solve_times)
    return out


# -------------------------------------------------------------------------------- policies
def make_dfranzen(token_scale: float = 62000.0, fade_fraction: float = 0.2, park_tokens: float | None = None,
                  action_weight: float = 0.25):
    def policy(g: GState, ctx: Ctx) -> float:
        tf = tail_fraction_at(ctx.now, ctx.horizon, fade_fraction)
        p = priority_value(g.k + 1, g.a_on, g.t_on, g.inst.n, tf, token_scale=token_scale,
                           action_weight=action_weight)
        if park_tokens is not None and g.t_on > park_tokens:
            p -= 100_000  # parked band: resumes only when no unparked game waits
        return p
    return policy


def fifo(g: GState, ctx: Ctx) -> float:
    return 0.0  # ties broken by arrival: round robin over trimmed games


class HazardIndex:
    """Gittins-style index: max over look-ahead tau of E[score gained within tau tokens] / E[tokens used],
    from the pooled KM hazard by level index (0..4+) with the constant tail, chaining fresh later levels.
    Uses only what the harness knows in submission mode: level index, level count, tokens on level."""

    B = 2000
    M = 200

    def __init__(self, data: dict, tail_start: float = 80_000.0, tail_scale: float = 1.0,
                 runs: list[str] | None = None):
        att = [a for a in data["attempts"] if runs is None or a["run"] in runs]
        B, M = self.B, self.M
        exposure = sum(a["tokens"] - tail_start for a in att if a["tokens"] > tail_start)
        events = sum(1 for a in att if a["tokens"] > tail_start and a["solved"])
        lam = tail_scale * events / exposure
        h_tail = 1 - math.exp(-lam * B)
        self.hz = {}
        self.sbar = {}
        for c in range(5):
            v = [(a["tokens"], a["solved"]) for a in att if min(a["level"], 4) == c]
            h = np.empty(M)
            for b in range(M):
                lo, hi = b * B, (b + 1) * B
                risk = sum(1 for t, _ in v if t > lo)
                ev = sum(1 for t, e in v if e and lo < t <= hi)
                h[b] = ev / risk if (risk >= 15 and lo < tail_start) else h_tail
            self.hz[c] = h
            sc = [min(115.0, (a["base"] / a["actions"]) ** 2 * 100) for a in att
                  if min(a["level"], 4) == c and a["solved"] and a["actions"] > 0]
            self.sbar[c] = float(np.mean(sc))
        self.fresh = {}
        for c, h in self.hz.items():
            s = np.concatenate([[1.0], np.cumprod(1 - h)])[:M]
            self.fresh[c] = s * h
        self.table = {}

    def _curve(self, n: int, k: int, b0: int) -> np.ndarray:
        """ratio(tau) for tau = 1..M bins, cumulative max taken by the caller."""
        key = (n, k, b0)
        if key in self.table:
            return self.table[key]
        B, M = self.B, self.M
        h = self.hz[min(k, 4)]
        hh = np.concatenate([h[b0:], np.full(b0, h[-1])])[:M]
        s = np.concatenate([[1.0], np.cumprod(1 - hh)])[:M]
        pmf = s * hh
        W = n * (n + 1) / 2
        reward = np.zeros(M)
        cur = pmf
        last_cdf = None
        for j in range(k, min(n, k + 5)):
            if j > k:
                cur = np.convolve(cur, self.fresh[min(j, 4)])[:M]
            cdf = np.cumsum(cur)
            reward += (j + 1) / W * self.sbar[min(j, 4)] * cdf
            last_cdf = cdf if j == n - 1 else None
        alive = 1 - last_cdf if last_cdf is not None else np.ones(M)
        used = np.cumsum(np.concatenate([[1.0], alive[:-1]])) * B
        ratio = np.maximum.accumulate(reward / used)
        self.table[key] = ratio
        return ratio

    def policy(self, g: GState, ctx: Ctx) -> float:
        b0 = min(self.M - 1, int(g.t_on // self.B))
        ratio = self._curve(g.inst.n, g.k, b0)
        tau = int(max(1, min(self.M, (ctx.horizon - ctx.now) * ctx.rate // self.B)))
        return float(ratio[tau - 1]) * 1e6


def oracle(g: GState, ctx: Ctx) -> float:
    """Knows every level's true cost and score: best cumulative score per token over the next levels
    still reachable before the deadline at this game's rate."""
    n, W = g.inst.n, g.inst.n * (g.inst.n + 1) / 2
    budget = max(0.0, ctx.horizon - ctx.now) * ctx.rate
    best, cum_r, cum_c = 0.0, 0.0, -g.t_on
    for j in range(g.k, n):
        cum_c += g.inst.cost[j]
        if not math.isfinite(cum_c) or cum_c > budget:
            break
        ls = min(115.0, (g.inst.base[j] / max(1, g.inst.acts[j])) ** 2 * 100)
        cum_r += (j + 1) / W * ls
        best = max(best, cum_r / max(1.0, cum_c))
    return best * 1e6
