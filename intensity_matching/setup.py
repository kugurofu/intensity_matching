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
        # launch
        ('share/' + package_name + '/launch', ['launch/map_matching.launch.xml','launch/make_globalmap.launch.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kugurofu',
    maintainer_email='tenten31569@icloud.com',
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
