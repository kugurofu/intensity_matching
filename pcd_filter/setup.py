from setuptools import find_packages, setup

package_name = 'pcd_filter'

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
        'pcd_filter = pcd_filter.pcd_filter:main',
        'obs_bayes_map = pcd_filter.obs_bayes_map:main',
        'bayes_map = pcd_filter.bayes_map:main',
        'bayes_map_average = pcd_filter.bayes_map_average:main',
        ],
    },
)
