import os
from setuptools import setup, find_namespace_packages

packages = find_namespace_packages(where="src", include=['ac_carrier_scenario.*'])

current_directory = os.path.dirname(os.path.realpath(__file__))
requirements_path = current_directory + '/requirements.txt'
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(
    name='AC_Carrier_Scenario',
    version='1.0',
    packages=packages,
    package_dir={'': 'src'},
    url='',
    license='',
    author='Jonathan Garza',
    author_email='',
    description='AI Project related to an AC Carrier Scenario',
    install_requires=install_requires
)
