from setuptools import setup, find_namespace_packages

packages = find_namespace_packages(where="src", include=['ac_carrier_scenario.*'])

setup(
    name='AC_Carrier_Scenario',
    version='1.0',
    packages=packages,
    package_dir={'': 'src'},
    url='',
    license='',
    author='Jonathan Garza',
    author_email='',
    description='AI Project related to an AC Carrier Scenario'
)
