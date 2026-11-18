"""Observable survival-forecast contracts and identity isolation."""
import copy
import json
import math
import unittest

import calibrate_tail as t


class TailForecastContract(unittest.TestCase):
    def test_rate_uses_censored_exposure(self):
        # At landmark10: event at30, event at60, censor at40. Exposure100,
        # events2. Dropping censoring or counting it as a solve fails this oracle.
        attempts = [{"tokens": x, "solved": solved, "level": 0}
                    for x, solved in [(30, True), (60, True), (40, False), (5, True)]]
        fit = t.rate_fit(attempts, 10)
        self.assertEqual(fit["events"], 2)
        self.assertEqual(fit["survivors"], 3)
        self.assertEqual(fit["exposure"], 100)
        self.assertEqual(fit["rate"], 0.02)

    def test_censoring_before_horizon_is_unknown(self):
        f = t.forecast_at_horizon(math.log(2) / 20_000, 10_000, False, 20_000)
        self.assertAlmostEqual(f["p"], 0.5)
        self.assertIsNone(f["observed_label"])
        self.assertEqual(t.forecast_at_horizon(1e-5, 10_000, True, 20_000)["observed_label"], 1)
        self.assertEqual(t.forecast_at_horizon(1e-5, 30_000, True, 20_000)["observed_label"], 0)
        self.assertEqual(t.forecast_at_horizon(1e-5, 20_000, False, 20_000)["observed_label"], 0)
        self.assertEqual(t.forecast_at_horizon(1e-5, 20_000, True, 20_000)["observed_label"], 1)

    def test_km_keeps_censoring_out_of_failure_count(self):
        rows = [{"residual_tokens": x, "observed_solve": solved}
                for x, solved in [(10, False), (20, True), (30, True)]]
        self.assertEqual(t.km_probability(rows, 10), 0)
        self.assertEqual(t.km_probability(rows, 20), 0.5)
        self.assertEqual(t.km_probability(rows, 30), 1)

    def test_future_revelation_cannot_change_heldout_forecast(self):
        data = json.loads((t.PARENT / "attempts.json").read_text())
        folds = t.h.folds_for(next(iter(data["games"].values())).keys())
        changed = copy.deepcopy(data)
        for a in changed["attempts"]:
            if a["game"] in folds[0] and a["tokens"] > 80_000:
                a["tokens"] = 80_000 + 3 * (a["tokens"] - 80_000)
                a["solved"] = not a["solved"]
        for cohort in ("c14_only", "all_runs_train_only_scaled"):
            before, fits_before = t.forecasts(data, cohort, 80_000)
            after, fits_after = t.forecasts(changed, cohort, 80_000)
            self.assertEqual(fits_before[0], fits_after[0])
            def predictions(rows):
                return {(r['run'], r['game'], r['level']): {m: {k: f['p'] for k, f in hs.items()}
                                                          for m, hs in r['forecast'].items()}
                        for r in rows if r['fold'] == 0}
            self.assertEqual(predictions(before), predictions(after))

    def test_stratified_prior_returns_pooled_rate_when_empty(self):
        rows = [{"tokens": x, "solved": solved, "level": 0}
                for x, solved in [(30, True), (60, True), (40, False)]]
        self.assertAlmostEqual(t.rate_fit(rows, 10, 4)["rate"], 0.02)


if __name__ == "__main__":
    unittest.main()
