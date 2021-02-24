from scipy.spatial.transform import Rotation

def euler_to_quaternion(roll, pitch, yaw):
    return (Rotation.from_euler('XYZ',(roll, pitch, yaw), degrees=False)).as_quat()