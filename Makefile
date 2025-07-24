# 🧪 Testing
test:
	@echo "🔍 Running unit tests..."
	pytest tests/

# 🧹 Linting
lint:
	@echo "🧼 Linting with flake8..."
	flake8 . --exclude=venv,__pycache__

# 🐳 Docker
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t pentest-suite .

docker-run:
	@echo "🚀 Running Docker container on port 5000..."
	docker run --rm -p 5000:5000 pentest-suite

# 🧪 Local CLI
cli:
	@echo "🖥️  Running CLI interface..."
	python cli.py ransomware_emulator --payload_name test_payload --file_count 5 --monitor

# 🌐 Local Web Server
app:
	@echo "🌐 Launching Flask app..."
	python app.py

# 📦 Packaging
package:
	@echo "📦 Building Python package..."
	python setup.py sdist bdist_wheel

# 🚀 Install in editable mode
install:
	pip install -e .

# 🧪 Run all checks
check: lint test

# 🧰 Clean builds
clean:
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -exec rm -r {} +

# 🧾 Help menu
help:
	@echo "📦 Available targets:"
	@echo "  make test          - Run unit tests"
	@echo "  make lint          - Run flake8 linter"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make cli           - Run example CLI command"
	@echo "  make app           - Run Flask app locally"
	@echo "  make package       - Build PyPI package"
	@echo "  make install       - Pip install in editable mode"
	@echo "  make check         - Run lint and test"
	@echo "  make clean         - Remove build artifacts"
