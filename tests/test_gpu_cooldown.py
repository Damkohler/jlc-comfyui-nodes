"""CPU-only behavioral tests; no ComfyUI or NVIDIA dependencies required."""
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
for name in ('prototype', 'prototype.nodes', 'prototype.nodes.util_nodes'):
    sys.modules.setdefault(name, types.ModuleType(name))
versions = types.ModuleType('prototype.jlc_custom_nodes_versions')
versions.JLC_UTIL_NODES_VERSION = 'test'
sys.modules[versions.__name__] = versions
spec = importlib.util.spec_from_file_location('prototype.nodes.util_nodes.cooldown', ROOT / 'nodes/util_nodes/jlc_gpu_cooldown.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def run_gate(self, temperatures=(), cancel_at=100, payload=None, **kwargs):
        clock = [0.0]
        readings = iter(temperatures)
        sensor = types.SimpleNamespace(index=0, read=lambda: next(readings), close=lambda: None)
        def interrupt():
            if clock[0] >= cancel_at:
                raise InterruptedError()
        payload = payload if payload is not None else [object(), {'samples': object()}]
        def sleep(seconds):
            clock[0] += seconds
        with patch.object(m.time, 'monotonic', lambda: clock[0]), patch.object(m, '_sleep', sleep), patch.object(m, '_interrupt', interrupt), patch.object(m, '_Sensor', lambda index: sensor):
            result = m.JLC_GPUCooldown().cooldown(payload, **kwargs)
        self.assertIs(result['result'][0], payload)
        return result

    def test_timer(self):
        result = self.run_gate(mode=['timer'], minimum_wait_seconds=[3])
        self.assertEqual(result['result'][1], 3)

    def test_disabled(self):
        result = self.run_gate(enabled=[False])
        self.assertEqual(result['result'][1], 0)

    def test_stability_resets(self):
        result = self.run_gate([69, 71, 69, 69, 69], mode='temperature', stable_seconds=4)
        self.assertEqual(result['result'][1], 8)

    def test_combined_minimum(self):
        result = self.run_gate([69]*6, minimum_wait_seconds=10, stable_seconds=2)
        self.assertEqual(result['result'][1], 10)

    def test_sensor_failure_holds_and_cancels(self):
        with self.assertRaises(InterruptedError):
            self.run_gate([], mode='temperature', cancel_at=6)

    def test_high_temperature_holds(self):
        with self.assertRaises(InterruptedError):
            self.run_gate([90]*5, mode='temperature', cancel_at=6)

    def test_never_cached(self):
        self.assertNotEqual(m.JLC_GPUCooldown.IS_CHANGED(), m.JLC_GPUCooldown.IS_CHANGED())

    def test_passthrough_preserves_list_and_item_identity(self):
        first = object()
        nested = {'samples': object()}
        payload = [first, nested]
        result = self.run_gate(mode='timer', minimum_wait_seconds=0, payload=payload)
        returned = result['result'][0]
        self.assertIs(returned, payload)
        self.assertIs(returned[0], first)
        self.assertIs(returned[1], nested)
        self.assertIs(returned[1]['samples'], nested['samples'])

    def test_wildcard_accepts_concrete_types(self):
        self.assertFalse(m.ANY_TYPE != 'IMAGE')
        self.assertFalse(m.ANY_TYPE != 'LATENT')
        self.assertFalse(m.ANY_TYPE != 'CUSTOM_TYPE')

    def test_sensor_read_time_counts_toward_combined_timer(self):
        clock = [0.0]

        def read():
            clock[0] += 2.0
            return 69.0

        sensor = types.SimpleNamespace(index=0, read=read, close=lambda: None)
        payload = [object()]
        with patch.object(m.time, 'monotonic', lambda: clock[0]), \
             patch.object(m, '_sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds)), \
             patch.object(m, '_interrupt', lambda: None), \
             patch.object(m, '_Sensor', lambda index: sensor):
            result = m.JLC_GPUCooldown().cooldown(
                payload,
                mode='combined',
                minimum_wait_seconds=1,
                stable_seconds=0,
            )
        self.assertEqual(result['result'][1], 2)

    def test_invalid_temperature_gpu_index_fails_configuration(self):
        with self.assertRaisesRegex(ValueError, 'GPU index'):
            m.JLC_GPUCooldown().cooldown([object()], mode='temperature', gpu_index=32)

    def test_unavailable_physical_gpu_fails_instead_of_retrying(self):
        sensor = types.SimpleNamespace(
            index=1,
            read=Mock(side_effect=m._GPUSelectionError('GPU index 1 is unavailable')),
            close=Mock(),
        )
        with patch.object(m, '_Sensor', return_value=sensor), \
             patch.object(m, '_interrupt', lambda: None):
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                m.JLC_GPUCooldown().cooldown(
                    [object()],
                    mode='temperature',
                    gpu_index=1,
                )
        sensor.close.assert_called()

    def test_timer_mode_does_not_use_gpu_selection(self):
        with patch.object(m, '_Sensor', side_effect=AssertionError('sensor should not initialize')), \
             patch.object(m, '_interrupt', lambda: None):
            result = m.JLC_GPUCooldown().cooldown(
                [object()],
                mode='timer',
                minimum_wait_seconds=0,
                gpu_index='unused',
            )
        self.assertEqual(result['result'][1], 0)

    def test_sleep_checks_cancellation_in_short_intervals(self):
        clock = [0.0]
        checks = []

        def interrupt():
            checks.append(clock[0])

        def sleep(seconds):
            clock[0] += seconds

        with patch.object(m.time, 'monotonic', lambda: clock[0]), \
             patch.object(m.time, 'sleep', sleep), \
             patch.object(m, '_interrupt', interrupt):
            m._sleep(1.0)
        self.assertGreaterEqual(len(checks), 5)
        self.assertTrue(all((right - left) <= 0.200001 for left, right in zip(checks, checks[1:])))

if __name__ == '__main__':
    unittest.main()
