#!usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node


class SpaceParam(Node):

    def __init__(self):
        super().__init__("space_param")
        self.L = [0.3375, 0.3993, 0.3993, 0.1260]

    def space_based_frame(self):

        w_list = [np.array([0.0, 0.0, 1.0]),
                  np.array([0.0, 1.0, 0.0]),
                  np.array([0.0, 0.0, 1.0]),
                  np.array([0.0, 1.0, 0.0]),
                  np.array([0.0, 0.0, 1.0]),
                  np.array([0.0, 1.0, 0.0]),
                  np.array([0.0, 0.0, 1.0])]

        q_list = [np.array([0.0, 0.0, self.L[0]]),
                  np.array([0.0, 0.0, self.L[0]]),
                  np.array([0.0, 0.0, self.L[0]+self.L[1]]),
                  np.array([0.0, 0.0, self.L[0]+self.L[1]]),
                  np.array([0.0, 0.0, self.L[0]+self.L[1]+self.L[2]]),
                  np.array([0.0, 0.0, self.L[0]+self.L[1]+self.L[2]]),
                  np.array([0.0, 0.0, self.L[0]+self.L[1]+self.L[2]+self.L[3]])]

        screw_list = []

        for i, w in enumerate(w_list):
            v = -np.cross(w, q_list[i])
            S = np.hstack((w, v))
            screw_list.append(S)

        S_list = np.column_stack(screw_list)

        return S_list


def main(args=None):
    rclpy.init(args=args)
    node = SpaceParam()
    rclpy.spin(node=node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
