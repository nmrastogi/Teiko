.PHONY: setup pipeline dashboard test

setup:
	pip install -r requirements.txt

pipeline:
	python load_data.py
	python analysis.py

dashboard:
	python server.py

test:
	pytest tests/test_data.py -v
