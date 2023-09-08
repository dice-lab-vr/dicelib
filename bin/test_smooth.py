#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from dicelib import ui
from dicelib.smoothing import spline_smooth
from dicelib.ui import ColoredArgParser
import numpy as np


def test_smoothing(streamline):
    smoothed = spline_smooth(streamline)
    print(smoothed)

def main():
    np.random.seed(0)
    # create 3D streamline with 10 random points
    streamline = np.random.rand(10, 3)
    print(streamline)
    print("\n\n")
    test_smoothing(streamline)

if __name__ == '__main__':
    main()
