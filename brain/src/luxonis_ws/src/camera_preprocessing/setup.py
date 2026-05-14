from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'camera_preprocessing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        # Copia i modelli ONNX nella cartella di installazione
        (os.path.join('lib', 'python3.12', 'site-packages', package_name),
            glob(os.path.join(package_name, '*.onnx'))),
    ],
    install_requires=['setuptools', 'onnxruntime', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='davide-pillon',
    maintainer_email='davide.pillon04@gmail.com',
    description='Camera preprocessing and ONNX lane inference for Luxonis OAK.',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'lane_inference_node = camera_preprocessing.luxonis_preprocessing:main',
        ],
    },
)
