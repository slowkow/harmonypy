.PHONY: clean all setup

all: clean
	python3 setup.py sdist bdist_wheel

setup: requirements.txt
	pip install -r requirements.txt

clean:
	rm -rf __pycache__ build *.egg-info .pytest_cache

