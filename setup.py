import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name                          = "harmonypy",
    version                       = "0.0.3",
    author                        = "Kamil Slowikowski",
    author_email                  = "kslowikowski@gmail.com",
    description                   = "A data integration algorithm.",
    long_description              = long_description,
    long_description_content_type = "text/markdown",
    url                           = "https://github.com/slowkow/harmonypy",
    packages                      = setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
