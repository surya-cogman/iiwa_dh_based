#!/usr/bin/env python3

import rclpy
import numpy as np
import matplotlib.pyplot as plt
from space_based import helper_fns
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class FourthJoint(Node):

    def __init__(self):
        super().__init__("fourth_joint")

        # storing actual_joints
        self.subscriber_ = self.create_subscription(
            JointState, '/joint_states', self.cb_joint_sub, 10)

        self.my_joints = ['joint1', 'joint2', 'joint3',
                          'joint4', 'joint5', 'joint6', 'joint7']
        self.joint_states = []
        self.joint_index = {}
        self.joint_index_initializer = False

        # link lenghths
        self.L = [0.3375, 0.3993, 0.3993, 0.1260]
        self.screw_list = self.space_screw()

        # defining transformation matrices

        self.T1 = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, self.L[0] +
                             self.L[1]+self.L[2]+self.L[3]],
                            [0.0, 0.0, 0.0, 1.0]])

        self.T2 = np.array([[0.0, 0.0, 1.0, (self.L[3])],
                            [0.0, 1.0, 0.0, 0.0],
                            [-1.0, 0.0, 0.0, self.L[0] + self.L[1] + self.L[2]],
                            [0.0, 0.0, 0.0, 1.0]])

        self.T3 = np.array([[0.0, 1.0, 0.0, 0.0],
                            [-1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, self.L[0] +
                             self.L[1]+self.L[2]+self.L[3]],
                            [0.0, 0.0, 0.0, 1.0]])

        # time variables

        self.t = 0.0
        self.dt = 0.01
        self.total_time = 10.0

        # initializing parameters

        self.T_prev = None
        self.q_prev = None
        self.q_history = []
        self.time_history = []

        # publisher and timer

        self.publisher_ = self.create_publisher(
            Float64MultiArray, '/position_controller/commands', 10)
        self.timer_ = self.create_timer(self.dt, self.cb_joint_publisher)

    def cb_joint_sub(self, msg):
        """
        Docstring for cb_joint_sub

        :param self: Description
        :param msg: Description
        """

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
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """

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

    def trajectory_generator(self, T_ini, T_fin, time, total_time, time_scale):
        """
        _summary_

        :param T_initial: _description_
        :type T_initial: _type_
        :param T_final: _description_
        :type T_final: _type_
        :param time: _description_
        :type time: _type_
        :param total_time: _description_
        :type total_time: _type_
        :param time_scale: _description_
        :type time_scale: _type_
        """

        u = min(time/total_time, 1.0)

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
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """

        if self.T_prev is None:
            self.T_prev = self.T_des
            return np.zeros(6)

        # delta_T = np.linalg.inv(self.T_prev)@self.T_des
        delta_T = self.T_des@np.linalg.inv(self.T_prev)

        twist = (1/self.dt) * \
            helper_fns.se3ToVec(helper_fns.MatrixLog6(delta_T))

        self.T_prev = self.T_des

        return twist

    def controller(self):
        """
        _summary_
        """

        self.t += self.dt

        self.T_des = self.trajectory_generator(
            self.T1, self.T2, self.t, self.total_time, "5")

        if len(self.joint_states) == 0:
            return None

        if self.q_prev is None:
            self.q_prev = self.joint_states

        self.V_des = self.twist_generator()

        Jac = helper_fns.JacobianSpace(self.screw_list, self.q_prev)

        jac_trans = Jac.transpose()

        pse_jac = np.linalg.inv((jac_trans@Jac + 0.001*np.eye(7)))@jac_trans

        q_dot = pse_jac@self.V_des

        q_cmd = self.q_prev + self.dt * q_dot
        self.q_prev = q_cmd

        return q_cmd

    def plot_results(self):

        q_hist = np.array(self.q_history)
        t_hist = np.array(self.time_history)

        plt.figure(figsize=(10, 6))
        for i in range(q_hist.shape[1]):
            plt.plot(t_hist, q_hist[:, i], label=f'joint{i+1}')

        plt.xlabel("Time [s]")
        plt.ylabel("Joint Angle [rad]")
        plt.title("Joint Trajectories (CLIK)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def cb_joint_publisher(self):
        """
        _summary_
        """

        joint_cmd = self.controller()

        if joint_cmd is None:
            return

        self.q_history.append(joint_cmd.copy())
        self.time_history.append(self.t)

        if self.t >= self.total_time:
            self.get_logger().info("trajectory completed")
            self.plot_results()
            rclpy.shutdown()
        # keep publishing final theta_cmd repeatedly

        msg = Float64MultiArray()
        msg.data = joint_cmd.tolist()
        self.publisher_.publish(msg)


def main(args=None):
    """
    _summary_

    :param args: _description_, defaults to None
    :type args: _type_, optional
    """
    rclpy.init(args=args)
    node = FourthJoint()
    rclpy.spin(node=node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
