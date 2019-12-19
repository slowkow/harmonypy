import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="harmonypy",
    version="0.0.1",
    author="Kamil Slowikowski",
    author_email="kslowikowski@gmail.com",
    description="A data integration method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slowkow/harmonypy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
