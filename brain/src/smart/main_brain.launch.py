from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='luxonis_ws',
            executable='traffic_sign_node',
            name='traffic_sign_node',
            output='screen',
        ),
        Node(
            package='brain',
            executable='main_brain',
            name='main_brain',
            output='screen',
        ),
    ])