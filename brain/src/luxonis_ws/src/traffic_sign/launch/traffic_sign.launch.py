import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('traffic_sign'),
        'config', 'params.yaml'
    )

    traffic_sign_node = Node(
        package='traffic_sign',
        executable='traffic_sign_node',
        name='traffic_sign_node',
        output='screen',
        parameters=[config],
    )

    return LaunchDescription([traffic_sign_node])
