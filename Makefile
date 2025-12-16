# X-ray Imaging Analysis - Makefile
# This Makefile provides convenient commands for development and deployment

.PHONY: help up build restart down logs clean status test lint format install dev update backup

# Default target - show available commands
help:
	@echo "X-ray Imaging Analysis - Available Commands:"
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  make up        - Start containers in detached mode"
	@echo "  make build     - Build and start containers (rebuild if needed)"
	@echo "  make restart   - Restart all containers"
	@echo "  make down      - Stop and remove containers"
	@echo "  make logs      - Show container logs (follow mode)"
	@echo "  make status    - Show container status"
	@echo ""
	@echo "🧹 Cleanup Commands:"
	@echo "  make clean     - Stop containers and remove volumes/networks"
	@echo "  make prune     - Remove unused Docker images and containers"
	@echo ""
	@echo "🔧 Development Commands:"
	@echo "  make install   - Install Python dependencies"
	@echo "  make dev       - Run Streamlit app locally (development mode)"
	@echo "  make test      - Run Python syntax checks"
	@echo "  make lint      - Run code linting"
	@echo "  make format    - Format Python code"
	@echo ""
	@echo "📦 Maintenance Commands:"
	@echo "  make update    - Git pull + rebuild containers"
	@echo "  make backup    - Create backup of project files"

# Docker commands
up:
	@echo "🚀 Starting X-ray Imaging Analysis containers..."
	docker compose up -d

build:
	@echo "🔨 Building and starting X-ray Imaging Analysis..."
	docker compose up -d --build

restart:
	@echo "🔄 Restarting containers..."
	docker compose restart

down:
	@echo "🛑 Stopping containers..."
	docker compose down

logs:
	@echo "📋 Showing container logs (Ctrl+C to exit)..."
	docker compose logs -f

status:
	@echo "📊 Container status:"
	docker compose ps

# Cleanup commands
clean:
	@echo "🧹 Cleaning up containers, volumes, and networks..."
	docker compose down --volumes --remove-orphans

prune:
	@echo "🗑️ Removing unused Docker resources..."
	docker system prune -f
	docker image prune -f

# Development commands
install:
	@echo "📦 Installing Python dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv for installation..."; \
		uv sync; \
	elif [ -f "requirements.txt" ]; then \
		echo "Using pip for installation..."; \
		pip install -r requirements.txt; \
	else \
		echo "No requirements.txt found, using pyproject.toml..."; \
		pip install -e .; \
	fi

dev:
	@echo "🔧 Starting Streamlit app in development mode..."
	@echo "Access the app at: http://localhost:8502"
	streamlit run menu_analyzer.py --server.port=8502 --server.enableCORS=true

test:
	@echo "🧪 Running Python syntax checks..."
	@echo "Checking menu_analyzer.py..."
	@python3 -m py_compile menu_analyzer.py && echo "✅ menu_analyzer.py: OK" || echo "❌ menu_analyzer.py: FAILED"
	@echo "Checking detector_conversion.py..."
	@python3 -m py_compile detector_conversion.py && echo "✅ detector_conversion.py: OK" || echo "❌ detector_conversion.py: FAILED"
	@echo "Checking uniformity.py..."
	@python3 -m py_compile uniformity.py && echo "✅ uniformity.py: OK" || echo "❌ uniformity.py: FAILED"
	@echo "Checking MTF.py..."
	@python3 -m py_compile MTF.py && echo "✅ MTF.py: OK" || echo "❌ MTF.py: FAILED"
	@echo "Checking NPS.py..."
	@python3 -m py_compile NPS.py && echo "✅ NPS.py: OK" || echo "❌ NPS.py: FAILED"

lint:
	@echo "🔍 Running code linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		echo "Running flake8..."; \
		flake8 *.py --max-line-length=120 --ignore=E501,W503; \
	else \
		echo "flake8 not installed, skipping lint check"; \
		echo "Install with: pip install flake8"; \
	fi

format:
	@echo "🎨 Formatting Python code..."
	@if command -v black >/dev/null 2>&1; then \
		echo "Running black formatter..."; \
		black *.py --line-length=120; \
	else \
		echo "black not installed, skipping formatting"; \
		echo "Install with: pip install black"; \
	fi

# Maintenance commands
update:
	@echo "📥 Updating X-ray Imaging Analysis..."
	@echo "Pulling latest changes from git..."
	git pull
	@echo "Rebuilding containers..."
	docker compose up -d --build
	@echo "✅ Update complete!"

backup:
	@echo "💾 Creating backup..."
	@BACKUP_NAME="xray-analysis-backup-$$(date +%Y%m%d-%H%M%S)" && \
	tar -czf "../$$BACKUP_NAME.tar.gz" \
		--exclude=".git" \
		--exclude="__pycache__" \
		--exclude=".venv" \
		--exclude="*.pyc" \
		. && \
	echo "✅ Backup created: ../$$BACKUP_NAME.tar.gz"

# Health check
health:
	@echo "🏥 Checking application health..."
	@if curl -f http://localhost:8502 >/dev/null 2>&1; then \
		echo "✅ Application is responding on port 8502"; \
	else \
		echo "❌ Application not responding on port 8502"; \
		echo "Try: make up"; \
	fi

# Quick development setup
setup:
	@echo "⚡ Setting up development environment..."
	make install
	make build
	make health
	@echo "🎉 Setup complete! Access the app at: http://localhost:8502"