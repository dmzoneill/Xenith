.PHONY: help install install-dev install-system-deps install-whisper install-python313 run test test-voice test-audio clean lint format check-deps shell env-info

# Default target
help:
	@echo "Xenith - Linux Desktop AI Agent"
	@echo ""
	@echo "Available targets:"
	@echo "  make install         - Install the package and dependencies (includes system deps)"
	@echo "  make install-system-deps - Install system dependencies (Python 3.13, cairo, etc.)"
	@echo "  make install-dev     - Install in development mode"
	@echo "  make install-whisper - Install OpenAI Whisper for wake word detection"
	@echo "  make run             - Run the main application"
	@echo "  make test          - Run test scripts"
	@echo "  make test-voice    - Run voice state test"
	@echo "  make test-audio    - Test audio device selection"
	@echo "  make clean         - Clean build artifacts and cache"
	@echo "  make lint          - Run linter (if available)"
	@echo "  make format        - Format code (if available)"
	@echo "  make check-deps    - Check if all dependencies are installed"
	@echo "  make shell          - Activate pipenv shell"
	@echo "  make env-info       - Show pipenv environment information"
	@echo "  make install-python313 - Install Python 3.13 (if not available)"
	@echo ""

# Install system dependencies (required for Python 3.13)
install-system-deps:
	@echo "Installing system dependencies for Python 3.13..."
	@if command -v dnf >/dev/null 2>&1; then \
		echo "Installing Python 3.13 and system packages..."; \
		sudo dnf install -y python3.13 python3.13-devel python3-cairo python3-gobject python3-atspi portaudio-devel || \
		(echo "⚠ Some packages failed to install. Continuing anyway..." && true); \
		echo "✓ System dependencies installed"; \
	else \
		echo "dnf not found. Please install manually:"; \
		echo "  sudo dnf install python3.13 python3.13-devel python3-cairo python3-gobject python3-atspi portaudio-devel"; \
	fi

# Install dependencies and package (using pipenv)
install: install-system-deps
	@echo ""
	@echo "Setting up pipenv environment with Python 3.13..."
	@if ! command -v pipenv >/dev/null 2>&1; then \
		echo "pipenv not found. Installing pipenv..."; \
		pip3 install pipenv; \
	fi
	@echo "Checking for Python 3.13..."
	@if ! command -v python3.13 >/dev/null 2>&1; then \
		echo ""; \
		echo "⚠ Python 3.13 not found. Installing via dnf..."; \
		if command -v dnf >/dev/null 2>&1; then \
			echo "Installing python3.13 and python3.13-devel..."; \
			sudo dnf install -y python3.13 python3.13-devel || (echo "Failed to install Python 3.13. Please run: sudo dnf install python3.13 python3.13-devel" && exit 1); \
			echo "✓ Python 3.13 installed"; \
		else \
			echo "dnf not found. Please install Python 3.13 manually:"; \
			echo "  sudo dnf install python3.13 python3.13-devel"; \
			echo "Or run: make install-python313"; \
			exit 1; \
		fi \
	fi
	@echo "Creating pipenv environment with Python 3.13..."
	@echo "Note: Some packages (cairo, pyatspi, python-dbus) are installed as system packages."
	@echo ""
	pipenv install --python 3.13 || (echo "" && echo "⚠ Dependency resolution failed. Trying with --skip-lock..." && pipenv install --python 3.13 --skip-lock)
	@echo "Installing project in editable mode..."
	@pipenv run pip install -e . || (echo "⚠ Editable install failed, trying alternative method..." && pipenv run python3 -m pip install -e . || echo "⚠ Could not install in editable mode, but dependencies are installed")
	@echo ""
	@echo "✓ Installation complete!"
	@echo "Note: Whisper (local voice-to-text) is included and will download models on first use."
	@echo ""
	@echo "To activate the environment, run: pipenv shell"
	@echo "Or run commands with: pipenv run <command>"
	@echo ""

# Install in development mode
install-dev: install
	@echo "Development mode installed"
	@echo "To install dev dependencies, add them to [dev-packages] in Pipfile and run: pipenv install --dev"

# Install Whisper for wake word detection
install-whisper:
	@echo "Installing OpenAI Whisper for wake word detection..."
	@echo "This may take a few minutes as it downloads the model..."
	pipenv install openai-whisper
	@echo ""
	@echo "✓ Whisper installed!"
	@echo "Note: The first time you run the app, Whisper will download a model (~150MB)"
	@echo ""
	@echo "For GPU acceleration, install PyTorch with CUDA support:"
	@echo "  pipenv run pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
	@echo "  (Replace cu121 with your CUDA version: cu118, cu121, etc.)"

# Run the main application
run:
	pipenv run python3 -m src.main

# Run all tests
test: test-voice test-audio

# Run voice state test
test-voice:
	pipenv run python3 test_voice_states.py

# Test audio devices
test-audio:
	pipenv run python3 test_audio_devices.py

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	@echo "Cleaned build artifacts and cache"

# Run linter (if available)
lint:
	@if command -v pylint >/dev/null 2>&1; then \
		pylint src/ --disable=C,R || true; \
	else \
		echo "pylint not installed. Install with: pip3 install pylint"; \
	fi
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 src/ --max-line-length=120 --ignore=E203,W503 || true; \
	else \
		echo "flake8 not installed. Install with: pip3 install flake8"; \
	fi

# Format code (if available)
format:
	@if command -v black >/dev/null 2>&1; then \
		black src/ test_*.py || true; \
	else \
		echo "black not installed. Install with: pip3 install black"; \
	fi
	@if command -v isort >/dev/null 2>&1; then \
		isort src/ test_*.py || true; \
	else \
		echo "isort not installed. Install with: pip3 install isort"; \
	fi

# Check if all dependencies are installed
check-deps:
	@echo "Checking dependencies in pipenv environment..."
	@pipenv run python3 -c "import sys; required = ['gi', 'cairo', 'numpy', 'sounddevice']; missing = []; \
		for pkg in required: \
			try: \
				if pkg == 'gi': \
					import gi; gi.require_version('Gtk', '4.0'); \
				elif pkg == 'cairo': \
					import cairo; \
				else: \
					__import__(pkg); \
				print(f'  ✓ {pkg}'); \
			except ImportError: \
				print(f'  ✗ {pkg} (missing)'); \
				missing.append(pkg); \
		sys.exit(0 if not missing else 1)" || \
		(echo "Some dependencies are missing. Run 'make install' to install them." && exit 1)
	@echo "All core dependencies are installed!"

# Development helpers
dev-setup: install-dev
	@echo "Development environment ready!"

# Quick run (no device selection prompt)
run-quick:
	@echo "Running Xenith (quick mode - using default audio device)..."
	@DEVICE_AUTO=1 pipenv run python3 -m src.main

# Activate pipenv shell
shell:
	@echo "Activating pipenv shell..."
	@echo "Run 'exit' to leave the shell"
	pipenv shell

# Show pipenv environment info
env-info:
	@echo "Pipenv Environment Info:"
	@pipenv --venv || echo "No virtual environment created yet. Run 'make install' first."
	@echo ""
	@pipenv --python || echo "Python version not set"

# Install Python 3.13
install-python313:
	@echo "Installing Python 3.13..."
	@./install-python313.sh

