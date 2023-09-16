install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C main.py

# test:
# 	pytest -vv --cov-report term-missing --cov=app test_*.py

format:
	black *.py

# all: install lint test format
all: install lint format