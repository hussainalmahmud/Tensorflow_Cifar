install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C main.py

test:
	pytest -vv --cov-report term-missing --cov=main tests/

format:
	black *.py

all: install lint format test
# all: install lint format