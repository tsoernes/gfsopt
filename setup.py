import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gfsopt",
    version="0.0.1",
    author=u"Torstein Sørnes",
    author_email="t.soernes@gmail.com",
    description="Scaffolding for the Global Function Search optimizer from Dlib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsoernes/gfsopt",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    python_requires='>=3.6',
    install_requires=['numpy', 'dlib', 'datadiff'],
)
