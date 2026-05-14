from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    args = [
        DeclareLaunchArgument('desired_speed', default_value='0.3'),
        DeclareLaunchArgument('debug_view', default_value='true'),
    ]

    sim_pkg_share = get_package_share_directory('sim_pkg')
    simulator = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(sim_pkg_share, 'launch', 'map_with_car.launch')
        )
    )

    lane_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='camera_preprocessing',
                executable='lane_inference_node',
                name='lane_inference_node',
                output='screen',
                prefix='/home/davide-pillon/BFMC/luxonis_ws/venv/bin/python3',
                parameters=[{
                    'input_topic':   '/oak/rgb/image_raw',
                    'command_topic': '/automobile/command',
                    'desired_speed': LaunchConfiguration('desired_speed'),
                    'debug_view':    LaunchConfiguration('debug_view'),
                }],
            )
        ],
    )

    return LaunchDescription(args + [simulator, lane_node])
