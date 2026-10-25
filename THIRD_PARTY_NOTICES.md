# Third-party components

This repository contains the PIPER CRAFT application. External SDKs, ROS workspaces, robot URDFs/meshes and dependency source trees are not bundled. Their original licenses continue to apply; this project's license does not relicense those components.

- Piper SDK: <https://github.com/agilexrobotics/piper_sdk>. The original installation used `piper_sdk` 0.6.1; its package metadata declares MIT.
- Pika ROS and robot description: <https://github.com/agilexrobotics/pika_ros> and <https://github.com/agilexrobotics/PikaAnyArm>. Configure separately supplied URDF/package paths. Do not assume the application license covers robot assets.
- Robot geometry / optimization environment: Pinocchio, HPP-FCL, CasADi/IPOPT, NumPy and SciPy; install separately and retain their notices.
- Desktop / RGB / video environment: dependency names and resolved versions are in `pyproject.toml` and `uv.lock`; these packages retain their own licenses.
- ROS Noetic, `rospy`, `sensor_msgs` and `cv_bridge` come from the operator's ROS installation.

If distributing a binary or container with these dependencies, include the notices required by the actual components distributed.
