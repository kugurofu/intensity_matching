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
        'static_intensity_map = intensity_matching.static_intensity_map:main',
        'reflection_intensity_map_multi_sub = intensity_matching.reflection_intensity_map_multi_sub:main',
        'multimap_matching = intensity_matching.multimap_matching:main',
        'rgb_map_matching = intensity_matching.rgb_map_matching:main',
        'png_map_matching = intensity_matching.png_map_matching:main',
        'reflection_intensity_localmap_multi_sub = intensity_matching.reflection_intensity_localmap_multi_sub:main',
        'reflection_intensity_globalmap_multi_sub = intensity_matching.reflection_intensity_globalmap_multi_sub:main',
        ],
    },
)
