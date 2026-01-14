#!/usr/bin/env python3

import rclpy
import numpy as np
import matplotlib.pyplot as plt

from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from space_based import helper_fns


class CLIKController(Node):

    def __init__(self):
        super().__init__("clik_controller")

        # ---------------- ROS ----------------
        self.sub = self.create_subscription(
            JointState, "/joint_states", self.joint_cb, 10
        )
        self.pub = self.create_publisher(
            Float64MultiArray, "/position_controller/commands", 10
        )

        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.control_loop)

        # ---------------- Robot ----------------
        self.joint_names = [
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6", "joint7"
        ]

        self.L = [0.3375, 0.3993, 0.3993, 0.1260]
        self.S = self.compute_space_screws()

        self.M = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, sum(self.L)],
            [0, 0, 0, 1]
        ])

        # ---------------- Trajectory ----------------
        self.T_start = self.M
        self.T_goal = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, sum(self.L)],
            [0, 0, 0, 1]
        ])

        self.t = 0.0
        self.T_des_prev = None

        # ---------------- State ----------------
        self.q_meas = None
        self.q_ref = None

        # ---------------- Gains ----------------

        self.Kp = 0.5*np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.64, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.54, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.44]])
        self.damping = 1e-3

        # ---------------- Logging ----------------
        self.q_log = []
        self.t_log = []

    # =========================================================
    # Callbacks
    # =========================================================

    def joint_cb(self, msg):
        q = np.zeros(7)
        for i, name in enumerate(self.joint_names):
            q[i] = msg.position[msg.name.index(name)]
        self.q_meas = q

        if self.q_ref is None:
            self.q_ref = q.copy()

    # =========================================================
    # Kinematics
    # =========================================================

    def compute_space_screws(self):
        w = [
            [0, 0, 1], [0, 1, 0], [0, 0, 1],
            [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]
        ]

        z = np.cumsum([self.L[0], 0, self.L[1], 0, self.L[2], 0, self.L[3]])
        q = [[0, 0, zi] for zi in z]

        S = []
        for wi, qi in zip(w, q):
            wi = np.array(wi)
            qi = np.array(qi)
            vi = -np.cross(wi, qi)
            S.append(np.hstack((wi, vi)))

        return np.column_stack(S)

    # =========================================================
    # Trajectory
    # =========================================================

    def desired_pose(self, t):
        u = min(t / 10.0, 1.0)
        s = 10*u**3 - 15*u**4 + 6*u**5

        return self.T_start @ helper_fns.MatrixExp6(
            helper_fns.MatrixLog6(
                np.linalg.inv(self.T_start) @ self.T_goal
            ) * s
        )

    def feedforward_twist(self, T_des):
        if self.T_des_prev is None:
            self.T_des_prev = T_des
            return np.zeros(6)

        delta = T_des @ np.linalg.inv(self.T_des_prev)
        Vd = (1 / self.dt) * helper_fns.se3ToVec(
            helper_fns.MatrixLog6(delta)
        )

        self.T_des_prev = T_des
        return Vd

    # =========================================================
    # Control
    # =========================================================

    def control_loop(self):
        if self.q_meas is None:
            return

        self.t += self.dt

        # Desired pose and twist
        T_des = self.desired_pose(self.t)
        Vd = self.feedforward_twist(T_des)

        # Actual pose
        T_act = helper_fns.FKinSpace(self.M, self.S, self.q_meas)

        # Error twist
        X_err = helper_fns.se3ToVec(
            helper_fns.MatrixLog6(T_des @ np.linalg.inv(T_act))
        )

        # CLIK twist
        V = Vd + self.Kp @ X_err

        # Jacobian
        J = helper_fns.JacobianSpace(self.S, self.q_meas)
        J_pinv = np.linalg.inv(
            J.T @ J + self.damping * np.eye(7)
        ) @ J.T

        q_dot = J_pinv @ V

        # Integrate INTERNAL reference
        self.q_ref = self.q_ref + self.dt * q_dot

        # Publish
        msg = Float64MultiArray()
        msg.data = self.q_ref.tolist()
        self.pub.publish(msg)

        # Log
        self.q_log.append(self.q_ref.copy())
        self.t_log.append(self.t)

        if self.t >= 10.0:
            self.plot()
            rclpy.shutdown()

    # =========================================================
    # Plot
    # =========================================================

    def plot(self):
        q = np.array(self.q_log)
        t = np.array(self.t_log)

        plt.figure(figsize=(10, 6))
        for i in range(7):
            plt.plot(t, q[:, i], label=f"joint{i+1}")
        plt.legend()
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Joint angle [rad]")
        plt.title("CLIK Joint Trajectories")
        plt.show()


def main():
    rclpy.init()
    node = CLIKController()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
