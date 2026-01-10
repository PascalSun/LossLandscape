#!/bin/bash
# Unified linting and formatting script for loss_landscape package
# Usage:
#   ./lint.sh          # Format code with black and isort, then check with flake8
#   ./lint.sh --check  # Only check code without formatting

set -e

CHECK_ONLY=false
if [[ "$1" == "--check" ]]; then
    CHECK_ONLY=true
fi

if [[ "$CHECK_ONLY" == true ]]; then
    echo "🔍 Checking code formatting and style..."
    echo ""
    echo "Running black (check only)..."
    black --check loss_landscape/
    
    echo "Running isort (check only)..."
    isort --check-only loss_landscape/
    
    echo "Running flake8..."
    flake8 loss_landscape/
    
    echo ""
    echo "✅ All checks passed!"
else
    echo "🎨 Formatting code..."
    echo ""
    echo "Running black..."
    black loss_landscape/
    
    echo "Running isort..."
    isort loss_landscape/
    
    echo ""
    echo "Running flake8..."
    flake8 loss_landscape/
    
    echo ""
    echo "✅ Formatting and linting complete!"
fi
