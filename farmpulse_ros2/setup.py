from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'farmpulse_ros2'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        # Install marker file for package index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # Install package.xml
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*_launch.py'))),
        # Install config files
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
        # Install message files
        (os.path.join('share', package_name, 'msg'),
            glob(os.path.join('msg', '*.msg'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Farmpulse ROS2 package for farm intruder detection using YOLO',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'farmpulse_node = farmpulse_ros2.farmpulse_node:main',
            'intruder_monitor = farmpulse_ros2.intruder_monitor:main',
        ],
    },
)