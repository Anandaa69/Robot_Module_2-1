import time
from robomaster import robot
from robomaster import camera


if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    ep_camera = ep_robot.camera
    ep_gimbal = ep_robot.gimbal
    ep_gimbal.recenter().wait_for_completed()
    ep_gimbal.move(pitch=-5,yaw=0).wait_for_completed()
    # 显示十秒图传
    ep_camera.start_video_stream(display=True, resolution=camera.STREAM_720P)
    time.sleep(10)
    ep_camera.stop_video_stream()

    ep_robot.close()

