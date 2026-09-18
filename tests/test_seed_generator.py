"""Backend seed contract tests; no ComfyUI frontend is required."""
import importlib.util
import math
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for name in ('prototype', 'prototype.nodes', 'prototype.nodes.util_nodes'):
    sys.modules.setdefault(name, types.ModuleType(name))
versions = sys.modules.setdefault(
    'prototype.jlc_custom_nodes_versions',
    types.ModuleType('prototype.jlc_custom_nodes_versions'),
)
versions.JLC_UTIL_NODES_VERSION = 'test'
spec = importlib.util.spec_from_file_location(
    'prototype.nodes.util_nodes.seed_generator',
    ROOT / 'nodes/util_nodes/jlc_seed_generator.py',
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Tests(unittest.TestCase):
    def test_frontend_schema_allows_unseeded_sentinel(self):
        seed_config = m.JLC_SeedGenerator.INPUT_TYPES()['required']['seed'][1]
        self.assertEqual(seed_config['min'], -1)
        self.assertTrue(seed_config['control_after_generate'])

    def test_unseeded_sentinel_survives_both_outputs(self):
        result = m.JLC_SeedGenerator().generator(
            seed=-1,
            randomize_replay='record',
        )
        self.assertEqual(result['result'], ({'seed': -1}, -1))
        self.assertEqual(result['ui']['jlc_seed'], ['-1'])
        self.assertEqual(result['ui']['jlc_seed_state'], ['unseeded'])

    def test_seed_validation_preserves_only_supported_range(self):
        self.assertEqual(m.JLC_SeedGenerator._coerce_seed(-1), -1)
        self.assertEqual(m.JLC_SeedGenerator._coerce_seed(-2), 0)
        self.assertEqual(m.JLC_SeedGenerator._coerce_seed(m.MAX_SEED + 1), m.MAX_SEED)

    def test_unseeded_is_volatile_and_numeric_seeds_are_cacheable(self):
        self.assertTrue(math.isnan(m.JLC_SeedGenerator.IS_CHANGED(seed=-1)))
        self.assertEqual(m.JLC_SeedGenerator.IS_CHANGED(seed=123), 123)

    def test_malformed_replay_mode_does_not_change_seed(self):
        result = m.JLC_SeedGenerator().generator(seed=42, randomize_replay='invalid')
        self.assertEqual(result['result'], ({'seed': 42}, 42))
        self.assertEqual(result['ui']['jlc_replay_mode'], ['off'])


if __name__ == '__main__':
    unittest.main()
