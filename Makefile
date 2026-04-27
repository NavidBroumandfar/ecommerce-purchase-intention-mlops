PYTHON ?= python3

.PHONY: install train evaluate test api docker-build docker-run clean

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m src.models.train

evaluate:
	$(PYTHON) -m src.models.evaluate

test:
	$(PYTHON) -m pytest

api:
	$(PYTHON) -m uvicorn src.api.main:app --reload

docker-build:
	docker build -t ecommerce-purchase-intention-mlops .

docker-run:
	docker run -p 8000:8000 ecommerce-purchase-intention-mlops

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	rm -f artifacts/model.joblib artifacts/metrics.json artifacts/model_metadata.json
	rm -f reports/drift_report.json
