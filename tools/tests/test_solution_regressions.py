"""Focused numerical checks for the rewritten attention implementation."""

from pathlib import Path
import runpy
import unittest

import numpy as np
import torch
from torch.nn import functional as F

DIRECTORY = Path(__file__).resolve().parents[2] / "app/content/solutions"


class AttentionTests(unittest.TestCase):
    def test_multihead_matches_torch_with_unequal_sequence_lengths(self):
        function = runpy.run_path(str(DIRECTORY / "q316.py"))["multi_head_attention"]
        rng = np.random.default_rng(21)
        for heads in (1, 2, 4):
            queries = rng.normal(size=(3, 8))
            keys = rng.normal(size=(5, 8))
            values = rng.normal(size=(5, 8))

            def split(array):
                return torch.from_numpy(
                    array.reshape(len(array), heads, 8 // heads).transpose(1, 0, 2)
                )

            expected = F.scaled_dot_product_attention(
                split(queries), split(keys), split(values)
            )
            expected = expected.transpose(0, 1).reshape(3, 8).numpy()
            np.testing.assert_allclose(
                function(queries, keys, values, heads), expected, atol=1e-12
            )

    def test_invalid_heads_raise_value_error(self):
        function = runpy.run_path(str(DIRECTORY / "q316.py"))["multi_head_attention"]
        for heads in (0, -1, 3, 1.5):
            with self.assertRaises(ValueError):
                function(np.ones((2, 4)), np.ones((2, 4)), np.ones((2, 4)), heads)

    def test_fully_masked_rows_are_zero_and_finite(self):
        function = runpy.run_path(str(DIRECTORY / "q309.py"))["q309"]
        query = np.array([[1000.0, 1000.0], [1.0, 2.0]])
        mask = np.array([[False, False], [True, False]])
        with np.errstate(all="raise"):
            output = function(query, query, np.eye(2), mask)
        np.testing.assert_array_equal(output, [[0, 0], [1, 0]])

    def test_normalized_difference_does_not_divide_by_zero(self):
        function = runpy.run_path(str(DIRECTORY / "q032.py"))["q032"]
        with np.errstate(all="raise"):
            output = function([2.0, 1.0], [-2.0, 3.0])
        np.testing.assert_array_equal(output, [0.0, -0.5])
