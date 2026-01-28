#!/bin/bash

echo "================================================"
echo "Stock Market RAG Chatbot - Setup Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "✓ All dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your API key!"
    echo "   Options:"
    echo "   - Groq: https://console.groq.com (recommended, free)"
    echo "   - Gemini: https://makersuite.google.com/app/apikey"
    echo ""
fi

# Test installation
echo "Testing installation..."
python3 -c "import gradio; import yfinance; import langchain; print('✓ All packages imported successfully')"

if [ $? -ne 0 ]; then
    echo "❌ Package import test failed"
    exit 1
fi

echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Add your API key to .env file:"
echo "   nano .env"
echo ""
echo "2. Run the application:"
echo "   python app.py"
echo ""
echo "Or run the demo (no API key needed):"
echo "   python demo_app.py"
echo ""
echo "================================================"
