#!/usr/bin/env python3

import rclpy
import numpy as np
from space_based import helper_fns
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class SpaceParam(Node):

    def __init__(self):
        super().__init__("space_param")

        # time paramters
        self.t = 0.0
        self.dt = 0.01
        self.total_time = 15.0

        # robot parameters
        self.L = [0.3375, 0.3993, 0.3993, 0.1260]
        self.S_list = self.space_screw()
        self.my_joints = ['joint1', 'joint2', 'joint3',
                          'joint4', 'joint5', 'joint6', 'joint7']

        self.started = False
        self.finished = False
        self.theta_hold = np.zeros(7)

        self.joint_states = []
        self.joint_index = {}
        self.joint_index_initializer = False

        # initializing custom parameters
        self.T_instant = np.eye(4)
        self.T_prev = None

        self.theta_cmd = None

        # transformation matrices to generate trajectory
        self.T_ini = np.array([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, self.L[0] +
                                   self.L[1]+self.L[2]+self.L[3]],
                               [0.0, 0.0, 0.0, 1.0]])

        self.T_fin = np.array([[0.0, 0.0, -1.0, self.L[3]],
                               [0.0, 1.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, self.L[0] +
                                   self.L[1]+self.L[2]],
                               [0.0, 0.0, 0.0, 1.0]])

        self.subscriber_ = self.create_subscription(
            JointState, '/joint_states', self.cb_joint_sub, 10)

        self.publisher_ = self.create_publisher(
            Float64MultiArray, '/position_controller/commands', 10)
        self.timer_ = self.create_timer(self.dt, self.joint_publisher)

    def cb_joint_sub(self, msg):

        if not self.joint_index_initializer:

            for joint_name in self.my_joints:
                if joint_name not in msg.name:
                    self.get_logger().error(
                        f"Missing joint {joint_name} in /joint_states"
                    )
                    return

                self.joint_index[joint_name] = msg.name.index(joint_name)

            self.joint_index_initializer = True

        theta = []

        for joint_name in self.my_joints:
            idx = self.joint_index[joint_name]
            theta.append(msg.position[idx])

        self.joint_states = np.array(theta)

    def space_screw(self):

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

        space_screw = np.column_stack(screw_list)

        return space_screw

    def trajectory_generator(self, T_ini, T_fin, time_step, total_time, time_scale):

        u = min(time_step/total_time, 1.0)

        if time_scale == "1":
            s = u
        elif time_scale == "3":
            s = 3*(u**2) - 2*(u**3)
        elif time_scale == "5":
            s = 10*(u**3) - 15*(u**4) + 6*(u**5)
        else:
            print(f"No time scale polynomial for given order {time_scale}")

        T = T_ini@helper_fns.MatrixExp6(
            helper_fns.MatrixLog6(np.linalg.inv(T_ini)@T_fin)*s)

        return T

    def twist_generator(self):

        if self.T_prev is None:
            self.T_prev = self.T_des
            return np.zeros(6)

        self.X_dot_des = (1/self.dt)*(helper_fns.se3ToVec(
            helper_fns.MatrixLog6(self.T_des@np.linalg.inv(self.T_prev))))

        self.T_prev = self.T_des

        return self.X_dot_des

    def controller(self):

        if len(self.joint_states) == 0:
            return None

        # -------- PHASE 1: BEFORE TRAJECTORY --------
        if not self.started:
            self.theta_hold = np.zeros(7)

            # wait one control cycle before starting trajectory
            self.started = True
            self.theta_cmd = self.theta_hold.copy()
            return self.theta_cmd

        # -------- PHASE 2: DURING TRAJECTORY --------
        if self.t < self.total_time:

            self.t += self.dt

            self.T_des = self.trajectory_generator(
                self.T_ini, self.T_fin, self.t, self.total_time, "5"
            )

            V_des = self.twist_generator()

            if self.theta_cmd is None:
                self.theta_cmd = self.joint_states.copy()

            J = helper_fns.JacobianSpace(self.S_list, self.theta_cmd)

            lambda2 = 1e-3
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda2 * np.eye(6))

            q_dot = J_pinv @ V_des
            self.theta_cmd = self.theta_cmd + self.dt * q_dot

            return self.theta_cmd

        # -------- PHASE 3: AFTER TRAJECTORY --------
        if not self.finished:
            self.theta_hold = self.theta_cmd.copy()
            self.finished = True
            self.get_logger().info("Trajectory completed. Holding final pose.")

        return self.theta_hold

    def joint_publisher(self):

        theta_cmd = self.controller()
        if theta_cmd is None:
            return

        if self.t >= self.total_time:
            self.get_logger().info("trajectory completed")
        # keep publishing final theta_cmd repeatedly

        msg = Float64MultiArray()
        msg.data = theta_cmd.tolist()
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpaceParam()
    rclpy.spin(node=node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
