"""
Unit tests for the crossworld MDP.

Yiqian Gan 2017
ganyq@umich.edu
"""

import unittest

import numpy as np
import numpy.random as rn

from . import crossworld


def make_random_crossworld():
    # grid_size = rn.randint(2, 15)
    wind = rn.uniform(0.0, 1.0)
    discount = rn.uniform(0.0, 1.0)
    return crossworld.Crossworld(discount,wind)


class TestTransitionProbability(unittest.TestCase):
    """Tests for Gridworld.transition_probability."""

    def test_sums_to_one(self):
        """Tests that the sum of transition probabilities is approximately 1."""
        # This is a simple fuzz-test.
        for _ in range(40):
            xw = make_random_gridworld()
            self.assertTrue(
                np.isclose(xw.transition_probability.sum(axis=2), 1).all(),
                'Probabilities don\'t sum to 1: {}'.format(xw))

    def test_manual_sums_to_one(self):
        """Tests issue #1 on GitHub."""
        xw = gridworld.Gridworld(0.3, 0.2)
        self.assertTrue(
            np.isclose(xw.transition_probability.sum(axis=2), 1).all())

if __name__ == '__main__':
    unittest.main()
