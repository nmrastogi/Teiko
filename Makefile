.PHONY: setup pipeline dashboard

setup:
	pip install -r requirements.txt

pipeline:
	python load_data.py
	python analysis.py

dashboard:
	python3 -m http.server 8080 --directory .
