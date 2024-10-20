# Commands for training the model
train:
	@echo "Running with setup_historical_config.sh"
	. ./setup_historical_config.sh && poetry run python src/training.py

test_preprocessing:
	@echo "Running with setup_historical_config.sh"
	. ./setup_historical_config.sh && poetry run pytest tests/test_apply_preprocessing.py

test_features_engineering:
	@echo "Running with setup_historical_config.sh"
	. ./setup_historical_config.sh && poetry run pytest tests/test_apply_features_engineering.py

# Commands for running the prediction script
predict:
	@echo "Running with setup_live_config.sh"
	. ./setup_live_config.sh && poetry run python src/prediction.py

# Commands for running the FastAPI app with Uvicorn
restapi:
	@echo "Running with setup_live_config.sh"
	. ./setup_live_config.sh && poetry run uvicorn src.restapi:app --host 0.0.0.0 --port 8000 --reload

# Health check for the REST API
check-health:
	curl -X GET http://localhost:8000/healthcheck

check-health-docker:
	curl -X GET http://localhost:5005/healthcheck

check-health-production:
	curl -X GET https://fuel-predictor-3f63c0-contactsouley-realtimemlfuelestimator-dev.deployments.quix.io/healthcheck


# Send a valid prediction request to the local API
request:
	curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"route": "Abidjan - Paris"}'

request-docker:
	curl -X POST http://localhost:5005/predict -H "Content-Type: application/json" -d '{"route": "Frankfurt - Berlin"}'

# Send a request to the production API (adjust the URL as needed)
request-production:
	curl -X POST https://fuel-predictor-3f63c0-contactsouley-realtimemlfuelestimator-dev.deployments.quix.io/predict -H "Content-Type: application/json" -d '{"route": "Frankfurt - Berlin"}'

# Send an invalid prediction request to test error handling
invalid-request:
	curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"route": "Niamey - Ouagadougou"}'

invalid-request-docker:
	curl -X POST http://localhost:5005/predict -H "Content-Type: application/json" -d '{"route": "Niamey - Ouagadougou"}'

# Build the Docker image
build:
	docker build -t contrail_estimator .

# Stop and remove existing container (if exists) before running a new one
clean:
	docker stop fuel_estimator || true && docker rm fuel_estimator || true

# Run the container after building
run: build
	docker run \
		--name fuel_estimator \
		--env-file ./setup_live_config.env \
		-p 5005:8000 \
		--name contrail_estimator \
		fuel_estimator

# Lint code with ruff
lint:
	poetry run ruff check --fix

# Format code with ruff
format:
	poetry run ruff format .

# Run both linting and formatting
lint-and-format: lint format