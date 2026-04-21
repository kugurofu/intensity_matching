from setuptools import find_packages, setup

package_name = 'intensity_matching'

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
        'intensity_map_beys = intensity_matching.intensity_map_beys:main',
        'reflection_intensity_map_beys = intensity_matching.reflection_intensity_map_beys:main',
        'reflection_intensity_map_discrete = intensity_matching.reflection_intensity_map_discrete:main',
        ],
    },
)
