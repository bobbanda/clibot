from setuptools import find_packages, setup

package_name = 'clibot_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                           'yolo = clibot_pkg.yolo:main',
                           'cam_sub = clibot_pkg.webcam:main',
                           'control = clibot_pkg.controller:main',
                           'database = clibot_pkg.fire_db:main'
                            
        ],
    },
)
