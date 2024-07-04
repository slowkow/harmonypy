We use [hatch] to develop harmonypy.

Copy the harmonypy code to your computer:

```
git clone https://github.com/slowkow/harmonpy
```

Then change to the newly created directory:

```
cd harmonypy
```

Install hatch:

```
pipx install hatch
```

Create a new environment just for harmonypy:

```
hatch env create
```

Once we have hatch and an environment, then we can enter a new shell:

```
hatch shell
```

In this environment, we can run tests:

```
hatch test
```

And we can also build the files needed for PyPI:

```
hatch build
```

We should double-check that the contents of the `.tar.gz` file do not include any files we do not want to publish:

```
tar tvf dist/harmonypy-0.0.10.tar.gz
-rw-r--r--  0 0      0          97 Feb  1  2020 harmonypy-0.0.10/harmonypy/__init__.py
-rw-r--r--  0 0      0       12783 Feb  1  2020 harmonypy-0.0.10/harmonypy/harmony.py
-rw-r--r--  0 0      0        4559 Feb  1  2020 harmonypy-0.0.10/harmonypy/lisi.py
-rw-r--r--  0 0      0        1824 Feb  1  2020 harmonypy-0.0.10/.gitignore
-rw-r--r--  0 0      0       35149 Feb  1  2020 harmonypy-0.0.10/LICENSE
-rw-r--r--  0 0      0        3126 Feb  1  2020 harmonypy-0.0.10/README.md
-rw-r--r--  0 0      0        1026 Feb  1  2020 harmonypy-0.0.10/pyproject.toml
```

When we're ready, we can publish to PyPI:

```
hatch publish
```

[hatch]: https://hatch.pypa.io

