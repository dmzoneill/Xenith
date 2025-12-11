#!/bin/bash
# Script to install Python 3.13 for Xenith project

set -e

echo "Installing Python 3.13 for Xenith..."

# Check if already installed
if command -v python3.13 >/dev/null 2>&1; then
    echo "✓ Python 3.13 is already installed: $(python3.13 --version)"
    exit 0
fi

# Check if running on Fedora
if command -v dnf >/dev/null 2>&1; then
    echo "Detected Fedora. Installing Python 3.13 from repositories..."
    
    # Install from Fedora repos
    if sudo dnf install -y python3.13 python3.13-devel; then
        echo "✓ Python 3.13 installed via dnf"
        echo "Version: $(python3.13 --version)"
        exit 0
    else
        echo "⚠ Failed to install Python 3.13 via dnf"
        echo ""
    fi
fi

# Check for pyenv
if command -v pyenv >/dev/null 2>&1; then
    echo "Found pyenv. Installing Python 3.13..."
    pyenv install 3.13.0
    pyenv local 3.13.0
    echo "✓ Python 3.13 installed via pyenv"
    exit 0
fi

# Install pyenv if not available
echo "pyenv not found. Installing pyenv..."
if [ -d "$HOME/.pyenv" ]; then
    echo "pyenv directory exists, updating..."
    cd ~/.pyenv && git pull
else
    echo "Installing pyenv..."
    curl https://pyenv.run | bash
    
    # Add to shell profile
    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q 'pyenv init' "$HOME/.bashrc"; then
            echo '' >> "$HOME/.bashrc"
            echo '# Pyenv' >> "$HOME/.bashrc"
            echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$HOME/.bashrc"
            echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> "$HOME/.bashrc"
            echo 'eval "$(pyenv init -)"' >> "$HOME/.bashrc"
        fi
    fi
    
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
fi

echo "Installing Python 3.13 via pyenv..."
echo "This may take 10-20 minutes..."
pyenv install 3.13.0

echo "Setting Python 3.13 as local version..."
pyenv local 3.13.0

echo ""
echo "✓ Python 3.13 installed successfully!"
echo ""
echo "Note: If you opened a new terminal, you may need to run:"
echo "  source ~/.bashrc"
echo "  cd $(pwd)"
echo ""
echo "Then run 'make install' to set up pipenv."

