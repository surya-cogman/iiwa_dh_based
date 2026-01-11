#!usr/bin/env python3

import math
import numpy as np


def t_matrix(a, alpha, d, theta):

    cos = math.cos
    sin = math.sin

    T = [[cos(theta), -sin(theta), 0, a],
         [sin(theta)*cos(alpha), cos(theta) *
          cos(alpha), -sin(alpha), -d*sin(alpha)],
         [sin(theta)*sin(alpha), cos(theta) *
          sin(alpha), cos(alpha), d*cos(alpha)],
         [0, 0, 0, 1]]

    return T
