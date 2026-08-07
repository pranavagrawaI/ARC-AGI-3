import importlib.util
import sys
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "reconstruction" / "transformations.py"
SPEC = importlib.util.spec_from_file_location("transformations", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TransformationInventoryTests(unittest.TestCase):
    def test_distinguishes_scale_from_tiling(self):
        source = MODULE.grid([[1, 2], [3, 4]])
        scaled = MODULE.nearest_scale(source, 2)
        tiled = MODULE.tiled(source, 2, 2)
        scale_results = {row.relation: row.matches for row in MODULE.evaluate(source, scaled)}
        tile_results = {row.relation: row.matches for row in MODULE.evaluate(source, tiled)}
        self.assertTrue(scale_results["integer_scale"])
        self.assertFalse(scale_results["tiled_repetition"])
        self.assertTrue(tile_results["tiled_repetition"])
        self.assertFalse(tile_results["integer_scale"])

    def test_congruence_and_colour_permutation_are_separate(self):
        source = MODULE.grid([[1, 2], [1, 1]])
        rotated = MODULE.rotate(source)
        recoloured = MODULE.grid([[7, 8], [7, 7]])
        self.assertTrue({r.relation: r.matches for r in MODULE.evaluate(source, rotated)}["dihedral_congruence"])
        self.assertTrue({r.relation: r.matches for r in MODULE.evaluate(source, recoloured)}["colour_permutation"])

    def test_embedded_histogram_and_algorithmic_progression(self):
        source = MODULE.grid([[1, 2]])
        target = MODULE.grid([[0, 0, 0], [1, 2, 0]])
        results = {row.relation: row.matches for row in MODULE.evaluate(source, target)}
        self.assertTrue(results["embedded_subpattern"])
        self.assertEqual(MODULE.algorithmic_progression([3, 0, 1, 2, 3]), ("advance_mod_4", 1))


if __name__ == "__main__":
    unittest.main()
