# Makefile for Sentenial-X AI repo
.PHONY: help venv install dev install-reqs train-ppo telemetry prefetch docker-build orchestrator eval clean

help:
	@echo "Sentenial-X AI Makefile"
	@echo "  make venv             # create virtualenv"
	@echo "  make install          # install package (dev extras)"
	@echo "  make install-reqs    # pip install -r requirements.txt"
	@echo "  make train-ppo       # train PPO (wrapper)"
	@echo "  make telemetry       # start telemetry consumer (foreground)"
	@echo "  make prefetch        # run prefetch models"
	@echo "  make docker-build    # build docker images"
	@echo "  make orchestrator    # run orchestrator demo"
	@echo "  make eval            # evaluate agent"
	@echo "  make clean           # cleanup pyc / __pycache__"

venv:
	python3 -m venv .venv
	@echo "Virtualenv created (.venv). Activate with: source .venv/bin/activate"

install: venv
	. .venv/bin/activate && pip install -e .[dev]

install-reqs: venv
	. .venv/bin/activate && pip install -r requirements.txt

train-ppo:
	@echo "Launching PPO training (wrapper script)..."
	./sentenial-x-ai/bin/train_ppo.sh

telemetry:
	@echo "Starting telemetry consumer..."
	./sentenial-x-ai/bin/start_telemetry_consumer.sh

prefetch:
	@echo "Running prefetch models script..."
	./sentenial-x-ai/bin/prefetch_models.sh

docker-build:
	@echo "Building docker images..."
	./sentenial-x-ai/bin/docker_build.sh

orchestrator:
	@echo "Running orchestrator demo..."
	./sentenial-x-ai/bin/run_orchestrator.sh

eval:
	@echo "Running evaluation..."
	./sentenial-x-ai/bin/eval_agent.sh

clean:
	@echo "Cleaning .pyc and __pycache__"
	find . -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
