#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst", encoding="utf-8") as history_file:
    history = history_file.read()

requirements = []

test_requirements = []

setup(
    author="JojoDevel",
    author_email="14841215+JojoDevel@users.noreply.github.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Python Boilerplate contains all the boilerplate you need to create a Python package.",
    entry_points={
        "console_scripts": [
            "depthai_lightning=depthai_lightning.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="depthai_lightning",
    name="depthai_lightning",
    packages=find_packages(include=["depthai_lightning", "depthai_lightning.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/JojoDevel/depthai_lightning",
    version="0.0.1",
    zip_safe=False,
)
