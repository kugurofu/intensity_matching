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
        'png_map_matching = intensity_matching.png_map_matching:main',
        'reflection_intensity_localmap_multi_sub = intensity_matching.reflection_intensity_localmap_multi_sub:main',
        'reflection_intensity_globalmap_multi_sub = intensity_matching.reflection_intensity_globalmap_multi_sub:main',
        'scoremap = intensity_matching.scoremap:main',
        ],
    },
)
