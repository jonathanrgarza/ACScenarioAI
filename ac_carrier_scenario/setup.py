from setuptools import setup

setup(
    name="AC_Carrier_Scenario",
    version="1.0",
    packages=["ac_carrier_scenario"],
    url="",
    license="",
    author="Jonathan Garza",
    author_email="",
    description="Package containing AC Carrier Scenario(s), including GYM environments for them.",
    requires=["gym~=0.19.0", "numpy~=1.21.4"]
)
