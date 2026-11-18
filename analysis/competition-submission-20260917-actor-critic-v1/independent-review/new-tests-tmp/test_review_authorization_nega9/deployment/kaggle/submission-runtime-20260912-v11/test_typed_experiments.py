import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "chassis"))

from decision_packet import packet, record_experiment


TARGETS = {
    "grid_changed": "any_change", "changed_bbox": "region", "shape_changed": "shape",
    "after_shape": "shape", "changed_cells": "change_count", "color_changes": "color",
    "state": "game_state", "levels_completed_delta": "progress",
}


def typed(metric, predictions, updates=None, otherwise="none of the predictions matched"):
    return {
        "target": TARGETS[metric],
        "metric": metric,
        "predictions": predictions,
        "updates": updates or {name: f"support {name} here" for name in predictions},
        "otherwise": otherwise,
    }


def make_packet(tmp_path, state, transitions):
    return packet(state, transitions, {}, {}, tmp_path, tmp_path)


def test_rejects_free_form_and_non_discriminating_experiments(tmp_path):
    state = {"turn": 0, "step": 0}
    with pytest.raises(ValueError, match="typed outcomes"):
        record_experiment(tmp_path, state, "where?", ["up", "left"], "UP",
                          {"expected": "up", "other": "anything"})
    with pytest.raises(ValueError, match="distinct"):
        record_experiment(tmp_path, state, "where?", ["up", "left"], "UP",
                          typed("changed_bbox", {"up": [0, 1, 1, 1], "left": [0, 1, 1, 1]}))
    with pytest.raises(ValueError, match="exactly match"):
        record_experiment(tmp_path, state, "where?", ["up", "left"], "UP",
                          typed("changed_bbox", {"up": [0, 1, 1, 1], "other region": [1, 0, 1, 1]}))
    assert not (tmp_path / "decision_experiments.jsonl").exists()


def test_target_metric_binding_is_structural_not_keyword_policing(tmp_path):
    state = {"turn": 0, "step": 0}
    with pytest.raises(ValueError, match="region target requires changed_bbox"):
        record_experiment(tmp_path, state, "Which region changes?",
                          ["center moves up", "center moves left"], "UP",
                          {**typed("grid_changed", {
                              "center moves up": True, "center moves left": False}),
                           "target": "region"})
    record = record_experiment(tmp_path, state, "Does UP change the shape?",
                               ["shape changes", "shape stays"], "UP",
                               typed("shape_changed", {"shape changes": True, "shape stays": False}))
    assert record["outcomes"]["target"] == "shape"


def test_scores_one_prediction_and_turns_both_sides_into_knowledge(tmp_path):
    state0 = {"turn": 0, "step": 0, "levels_completed": 0}
    record_experiment(tmp_path, state0, "which region?", ["moves up", "moves left"], "UP",
                      typed("changed_bbox", {
                          "moves up": [0, 1, 1, 1],
                          "moves left": [1, 0, 1, 1],
                      }))
    transition = {"turn": 0, "step": 1, "action": "UP",
                  "before": {"grid": [[0, 0, 0], [0, 7, 0], [0, 0, 0]]},
                  "after": {"grid": [[0, 0, 0], [7, 0, 0], [0, 0, 0]],
                            "state": "NOT_FINISHED", "levels_completed": 0}}
    result = make_packet(tmp_path, {"turn": 1, "step": 1}, [transition])
    scored = result["experiment_result"]
    assert scored["status"] == "matched"
    assert scored["observed"] == [1, 0, 1, 1]
    assert scored["matched_prediction_label"] == "moves left"
    assert scored["knowledge"] == [
        {"hypothesis_label": "moves up", "prediction": [0, 1, 1, 1],
         "matched_observation": False, "model_proposed_update": "support moves up here"},
        {"hypothesis_label": "moves left", "prediction": [1, 0, 1, 1],
         "matched_observation": True, "model_proposed_update": "support moves left here"},
    ]
    assert len((tmp_path / "experiment_results.jsonl").read_text().splitlines()) == 1
    make_packet(tmp_path, {"turn": 1, "step": 1}, [transition])
    assert len((tmp_path / "experiment_results.jsonl").read_text().splitlines()) == 1


def test_unpredicted_result_contradicts_all_predictions_and_preserves_observation(tmp_path):
    state0 = {"turn": 0, "step": 0, "levels_completed": 0}
    record_experiment(tmp_path, state0, "which region?", ["moves up", "moves left"], "UP",
                      typed("changed_bbox", {
                          "moves up": [0, 1, 1, 1],
                          "moves left": [1, 0, 1, 1],
                      }, otherwise="observed a third region; revise both mappings"))
    transition = {"turn": 0, "step": 1, "action": "UP",
                  "before": {"grid": [[0, 0, 0], [0, 7, 0], [0, 0, 0]]},
                  "after": {"grid": [[0, 0, 0], [0, 0, 0], [0, 0, 7]],
                            "state": "NOT_FINISHED", "levels_completed": 0}}
    scored = make_packet(tmp_path, {"turn": 1, "step": 1}, [transition])["experiment_result"]
    assert scored["status"] == "none_of_predictions"
    assert scored["observed"] == [1, 1, 2, 2]
    assert scored["matched_prediction_label"] is None
    assert not any(item["matched_observation"] for item in scored["knowledge"])
    assert scored["otherwise"] == "observed a third region; revise both mappings"


@pytest.mark.parametrize(
    "metric,predictions,expected",
    [
        ("grid_changed", {"changed": True, "unchanged": False}, True),
        ("shape_changed", {"reshaped": True, "same shape": False}, False),
        ("after_shape", {"two by two": [2, 2], "three by three": [3, 3]}, [2, 2]),
        ("changed_cells", {"two cells": 2, "zero cells": 0}, 2),
        ("state", {"continues": "NOT_FINISHED", "wins": "WIN"}, "NOT_FINISHED"),
        ("levels_completed_delta", {"no level": 0, "one level": 1}, 0),
        ("color_changes", {
            "swap 0 and 1": [{"before_color": 0, "after_color": 1, "count": 1},
                              {"before_color": 1, "after_color": 0, "count": 1}],
            "no color changes": [],
        }, [{"before_color": 0, "after_color": 1, "count": 1},
            {"before_color": 1, "after_color": 0, "count": 1}]),
    ],
)
def test_supported_metric_types_are_host_scored(tmp_path, metric, predictions, expected):
    hypotheses = list(predictions)
    record_experiment(tmp_path, {"turn": 0, "step": 0, "levels_completed": 4},
                      "what follows?", hypotheses, "UP", typed(metric, predictions))
    transition = {"turn": 0, "step": 1, "action": "UP",
                  "before": {"grid": [[0, 1], [0, 0]], "levels_completed": 4},
                  "after": {"grid": [[1, 0], [0, 0]], "state": "NOT_FINISHED",
                            "levels_completed": 4}}
    scored = make_packet(tmp_path, {"turn": 1, "step": 1}, [transition])["experiment_result"]
    assert scored["metric"] == metric
    assert scored["observed"] == expected
    assert scored["status"] == "matched"
