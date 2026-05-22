from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'acea_concert'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(where='src', exclude=['__pycache__']),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py') + glob('launch/*.py')),
        # Install config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.yml')),
        # Install rviz files
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.com',
    description='Trajectory optimization, simulation and replay for the CONCERT welding robot.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'controller = controller:main',
        ],
    },
)
