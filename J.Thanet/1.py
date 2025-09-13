# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from robomaster import robot
import time


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_servo = ep_robot.servo
    ep_gripper = ep_robot.gripper

    # 舵机3 转到0度
    ep_servo.moveto(index=1, angle=5).wait_for_completed()
    time.sleep(1)
    # ep_gripper.close(power = 50)

    ep_robot.close()




# import time
# import robomaster
# from robomaster import robot


# if __name__ == '__main__':
#     ep_robot = robot.Robot()
#     ep_robot.initialize(conn_type="ap")

#     ep_gripper = ep_robot.gripper

#     # 张开机械爪
#     ep_gripper.open(power=50)
#     time.sleep(1)
#     ep_gripper.pause()

#     # 闭合机械爪
#     ep_gripper.close(power=100)
#     time.sleep(1)
#     ep_gripper.pause()

#     ep_robot.close()