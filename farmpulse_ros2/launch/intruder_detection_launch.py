#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('farmpulse_ros2')
    
    # Path to config file
    config_file = os.path.join(pkg_dir, 'config', 'intruder_detection.yaml')
    
    # Declare launch arguments
    camera_source_arg = DeclareLaunchArgument(
        'camera_source',
        default_value='/dev/video0',
        description='Camera source (device path or RTSP URL)'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to configuration file'
    )
    
    model_size_arg = DeclareLaunchArgument(
        'model_size',
        default_value='yolov8n.pt',
        description='YOLO model size (n, s, m, l, x)'
    )
    
    save_detections_arg = DeclareLaunchArgument(
        'save_detections',
        default_value='true',
        description='Save detection images'
    )
    
    # Create the node
    farmpulse_node = Node(
        package='farmpulse_ros2',
        executable='farmpulse_node',
        name='farmpulse_intruder_detector',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'camera_source': LaunchConfiguration('camera_source'),
                'model_size': LaunchConfiguration('model_size'),
                'save_detections': LaunchConfiguration('save_detections'),
            }
        ],
        remappings=[
            # Add any topic remappings here if needed
        ]
    )
    
    # Log info
    log_info = LogInfo(
        msg=['Starting Farmpulse Intruder Detection System with camera: ', 
             LaunchConfiguration('camera_source')]
    )
    
    return LaunchDescription([
        camera_source_arg,
        config_file_arg,
        model_size_arg,
        save_detections_arg,
        log_info,
        farmpulse_node
    ])