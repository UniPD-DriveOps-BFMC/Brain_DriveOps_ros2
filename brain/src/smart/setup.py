from setuptools import setup
from glob import glob
import os

package_name = 'smart'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=['oak_camera_node'],   # expose top-level scripts as importable modules
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*.onnx')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='davide-pillon',
    maintainer_email='davide.pillon04@gmail.com',
    description='BFMC main brain and camera nodes',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'oak_camera_node = oak_camera_node:main',
        ],
    },
)
