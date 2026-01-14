#!/usr/bin/env python3
import rclpy
import numpy as np
import matplotlib.pyplot as plt

from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from space_based import helper_fns


class Controller(Node):
    def __init__(self):
        super().__init__("controller")

        # ---------------- Parameters ----------------
        self.dt = 0.01
        self.total_time = 10.0
        self.Kp = 5.0 * np.eye(6)
        self.damping = 1e-3

        # ---------------- ROS ----------------
        self.sub = self.create_subscription(
            JointState, "/joint_states", self.joint_cb, 10)
        self.pub = self.create_publisher(
            Float64MultiArray, "/position_controller/commands", 10)
        self.timer = self.create_timer(self.dt, self.control_loop)

        # ---------------- Robot ----------------
        self.joint_names = ["joint1", "joint2", "joint3",
                            "joint4", "joint5", "joint6", "joint7"]
        self.joint_index = None

        self.L = [0.3375, 0.3993, 0.3993, 0.1260]
        self.S = self.compute_space_screws()

        self.M = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, sum(self.L)],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # ---------------- Trajectory ----------------
        self.T_start = self.M
        self.T_goal = np.array([
            [0.0, 0.0, 1.0, self.L[2] + self.L[3]],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, self.L[0] + self.L[1]],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # ---------------- State ----------------
        self.t = 0.0
        self.T_des_prev = None
        self.q_meas = None
        self.q_ref = None
        self.hold_final = False   # flag to keep publishing final pose

        # ---------------- Logging ----------------
        self.q_log = []
        self.t_log = []

    # =========================================================
    # Callbacks
    # =========================================================
    def joint_cb(self, msg: JointState):

        if self.joint_index is None:
            if not set(self.joint_names).issubset(set(msg.name)):
                missing = [n for n in self.joint_names if n not in msg.name]
                self.get_logger().error(
                    f"Missing joints in /joint_states: {missing}")
                return
            self.joint_index = {name: msg.name.index(
                name) for name in self.joint_names}

        q = np.array([msg.position[self.joint_index[name]]
                     for name in self.joint_names], dtype=float)
        self.q_meas = q

        if self.q_ref is None:
            self.q_ref = q.copy()

    # =========================================================
    # Kinematics
    # =========================================================
    def compute_space_screws(self):
        w_list = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        z = np.cumsum([self.L[0], 0.0, self.L[1],
                      0.0, self.L[2], 0.0, self.L[3]])
        q_list = [np.array([0.0, 0.0, zi]) for zi in z]

        S_list = []
        for w, q in zip(w_list, q_list):
            v = -np.cross(w, q)
            S_list.append(np.hstack((w, v)))
        return np.column_stack(S_list)

    # =========================================================
    # Trajectory
    # =========================================================
    def desired_pose(self, t):
        u = min(t / self.total_time, 1.0)
        s = 10*u**3 - 15*u**4 + 6*u**5
        A = helper_fns.MatrixLog6(np.linalg.inv(self.T_start) @ self.T_goal)
        return self.T_start @ helper_fns.MatrixExp6(A * s)

    def feedforward_twist(self, T_des):

        if self.T_des_prev is None:
            self.T_des_prev = T_des
            return np.zeros(6)
        delta = T_des @ np.linalg.inv(self.T_des_prev)
        Vd = (1.0 / self.dt) * helper_fns.se3ToVec(helper_fns.MatrixLog6(delta))
        self.T_des_prev = T_des
        return Vd

    # =========================================================
    # Control
    # =========================================================
    def control_loop(self):

        if self.q_meas is None or self.q_ref is None:
            return

        # If trajectory is finished, just keep publishing final pose
        if self.hold_final:
            msg = Float64MultiArray()
            msg.data = self.q_ref.tolist()
            self.pub.publish(msg)
            return

        self.t += self.dt

        # Desired pose and twist
        T_des = self.desired_pose(self.t)
        Vd = self.feedforward_twist(T_des)

        # Actual pose
        T_act = helper_fns.FKinSpace(self.M, self.S, self.q_meas)

        # Error twist
        X_err = helper_fns.se3ToVec(
            helper_fns.MatrixLog6(T_des @ np.linalg.inv(T_act)))

        # CLIK twist
        V = Vd + self.Kp @ X_err

        # Jacobian
        J = helper_fns.JacobianSpace(self.S, self.q_meas)
        J_pinv = np.linalg.inv(J.T @ J + self.damping * np.eye(7)) @ J.T

        q_dot = J_pinv @ V

        # Integrate internal reference
        self.q_ref = self.q_ref + self.dt * q_dot

        # Publish
        msg = Float64MultiArray()
        msg.data = self.q_ref.tolist()
        self.pub.publish(msg)

        # Log
        self.q_log.append(self.q_ref.copy())
        self.t_log.append(self.t)

        # When trajectory time is up, switch to hold mode
        if self.t >= self.total_time:
            self.get_logger().info("Trajectory completed. Holding final pose.")
            self.plot()
            self.hold_final = True   # keep publishing final pose indefinitely

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
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Joint angle [rad]")
        plt.title("CLIK Joint Trajectories")
        plt.show()


def main():
    rclpy.init()
    node = Controller()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
