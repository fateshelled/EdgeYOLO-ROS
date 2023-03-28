from setuptools import setup

import os
from glob import glob

package_name = 'edgeyolo_ros_py'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('./launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='fateshelled',
    author_email="53618876+fateshelled@users.noreply.github.com",
    maintainer='fateshelled',
    maintainer_email="53618876+fateshelled@users.noreply.github.com",
    description='EdgeYOLO ROS 2 python',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'edgeyolo_onnx_node = '+package_name+'.edgeyolo_onnx_node:main',
        ],
    },
)

