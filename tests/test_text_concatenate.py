"""Prompt ordering and saved-workflow compatibility without a ComfyUI server."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('text_concatenate', ROOT / 'nodes/util_nodes/jlc_text_concatenate.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TextConcatenateTests(unittest.TestCase):
    def run_node(self, sections, count=None):
        return module.JLC_TextConcatenate().concatenate(json.dumps(sections), len(sections) if count is None else count)

    def test_legacy_strings_keep_order_and_individual_pins(self):
        result = self.run_node(['first', 'second'])
        self.assertEqual(result[:3], ('first\n\nsecond', 'first', 'second'))
        self.assertEqual(len(result), 33)

    def test_positive_then_negative_order_and_stable_ties(self):
        result = self.run_node([{'text': 'last', 'order': -1}, {'text': 'a', 'order': 2},
                    {'text': 'before-last', 'order': -2}, {'text': 'b', 'order': 2}, {'text': 'first', 'order': 1}])
        self.assertEqual(result[0], 'first\n\na\n\nb\n\nbefore-last\n\nlast')
        self.assertEqual(result[1:6], ('last', 'a', 'before-last', 'b', 'first'))

    def test_disabled_and_blank_sections_keep_independent_outputs(self):
        result = self.run_node([{'text': 'hidden', 'enabled': False}, '   ', 'visible'])
        self.assertEqual(result[:4], ('visible', 'hidden', '   ', 'visible'))

    def test_applied_count_does_not_route_unapplied_sections(self):
        self.assertEqual(self.run_node(['a', 'b'], 1)[:3], ('a', 'a', ''))

    def test_multiline_bytes_are_preserved(self):
        self.assertEqual(self.run_node(['a\nb', ' c '])[0], 'a\nb\n\n c ')

    def test_maximum_sections_and_default_output_contract(self):
        result = self.run_node([str(n) for n in range(32)])
        self.assertEqual(result[-1], '31')
        self.assertEqual(module.JLC_TextConcatenate.RETURN_NAMES[0], 'text')
        self.assertEqual(module.JLC_TextConcatenate.RETURN_NAMES[-1], 'section_32')
        self.assertEqual(list(module.JLC_TextConcatenate.INPUT_TYPES()['required']), ['sections', 'number_of_sections'])

    def test_invalid_json_types_and_limits_are_rejected(self):
        for value in ('bad', '{}', '[1]', '[{"text":2}]', '[{"text":"x","enabled":1}]',
                      '[{"order":0}]', '[{"order":true}]', '[{"order":1.5}]'):
            with self.subTest(value=value), self.assertRaises(ValueError):module.JLC_TextConcatenate().concatenate(value)
        for count in (0, 33):
            with self.assertRaises(ValueError):self.run_node(['a'], count)
        with self.assertRaises(ValueError):self.run_node(['a'] * 33)

    def test_node_is_registered_without_dropping_upstream_nodes(self):
        tree = ast.parse((ROOT / '__init__.py').read_text(encoding='utf-8'))
        mappings = next(n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'NODE_CLASS_MAPPINGS')
        keys = {k.value for k in mappings.keys if isinstance(k, ast.Constant)}
        self.assertTrue({'JLC_TextConcatenate', 'JLC_GPUCooldown', 'JLC_DynamicMultiReroute'} <= keys)


if __name__ == '__main__':unittest.main()
