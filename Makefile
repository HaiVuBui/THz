.PHONY: all test 
TARGET = exp.Experiment

all:
	poetry run python $(TARGET)

test:
	python -m test.py

