https://packaging.python.org/tutorials/packaging-projects/

Now run this command from the same directory where setup.py is located:

    python3 setup.py sdist bdist_wheel

This command should output a lot of text and once completed should generate two files in the dist directory:

    dist/
      example_pkg_YOUR_USERNAME_HERE-0.0.1-py3-none-any.whl
      example_pkg_YOUR_USERNAME_HERE-0.0.1.tar.gz

python3 -m pip install --user --upgrade twine

Run Twine to upload all of the archives under dist:

    python3 -m twine upload dist/harmonypy-0.0.4*
