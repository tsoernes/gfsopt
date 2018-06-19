import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ['numpy', 'dlib', 'datadiff']
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    install_requires = []

setuptools.setup(
    name="gfsopt",
    packages=["gfsopt"],
    version="1.0.1",
    author="Torstein Soernes",
    author_email="t.soernes@gmail.com",
    description="Scaffolding for the Global Function Search optimizer from Dlib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsoernes/gfsopt",
    project_urls={
        'Documentation': 'http://gfsopt.readthedocs.io/',
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    python_requires='>=3.6',
    install_requires=install_requires,
)
