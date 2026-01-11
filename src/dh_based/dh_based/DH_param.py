import rclpy
import numpy as np
import math
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from .helper_fns import t_matrix


class DhParam(Node):
    def __init__(self):
        super().__init__('DH_param')

        self.publisher_ = self.create_publisher(
            Float64MultiArray, '/ee_position', 10)

        self.subscriber_ = self.create_subscription(
            JointState, '/joint_states', self.cb_joint_sub, 10)

        self.my_joints = ['joint1', 'joint2', 'joint3',
                          'joint4', 'joint5', 'joint6', 'joint7']
        self.t = 0.0
        self.dt = 0.01
        self.l = [0.3375, 0.3993, 0.3993, 0.1260]

        self.theta = []
        self.joint_index = {}
        self.joint_index_initialize = False

        self.timer_ = self.create_timer(self.dt, self.cb_pub)

    def cb_joint_sub(self, msg):

        # Build mapping only once
        if not self.joint_index_initialize:

            self.joint_index = {}

            for joint_name in self.my_joints:
                if joint_name not in msg.name:
                    self.get_logger().error(
                        f"Missing joint {joint_name} in /joint_states"
                    )
                    return

                self.joint_index[joint_name] = msg.name.index(joint_name)

            self.joint_index_initialize = True

        # Read joint positions in FIXED orderNone
        theta = []
        for joint_name in self.my_joints:
            idx = self.joint_index[joint_name]
            theta.append(msg.position[idx])

        self.theta = np.array(theta)

    def cd_fk(self):

        if len(self.theta) < 7:
            return None

        dh = [[0, 0, self.l[0], self.theta[0]],
              [0, -1.5708, 0, self.theta[1]],
              [0, 1.5708, self.l[1], self.theta[2]],
              [0, -1.5708, 0, self.theta[3]],
              [0, 1.5708, self.l[2], self.theta[4]],
              [0, -1.5708, 0, self.theta[5]],
              [0, 1.5708, self.l[3], self.theta[6]]]
        dh = np.array(dh)
        r, c = dh.shape

        self.T_all = []
        T = np.eye(4)
        for i in range(r):
            T = T @ t_matrix(a=dh[i][0], alpha=dh[i][1],
                             d=dh[i][2], theta=dh[i][3])

            self.T_all.append(T)

        self.Tbase_ee = T

        return T

    def cb_pub(self):

        T = self.cd_fk()

        if T is None:
            return
        msg = Float64MultiArray()
        msg.data = T.flatten().tolist()
        self.publisher_.publish(msg=msg)

        self.get_logger().info(f"T : {T}")


def main(args=None):
    rclpy.init(args=args)
    node = DhParam()
    rclpy.spin(node=node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
