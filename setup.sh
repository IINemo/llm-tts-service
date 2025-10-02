#!/bin/bash
# Setup script for llm-tts-service
# Installs package dependencies and lm-polygraph dev branch

set -e  # Exit on error

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LM_POLYGRAPH_DIR="$SCRIPT_DIR/lm-polygraph"

# Parse arguments
UPDATE_ONLY=false
if [ "$1" = "--update" ] || [ "$1" = "-u" ]; then
    UPDATE_ONLY=true
fi

install_lm_polygraph() {
    echo -e "${YELLOW}Setting up lm-polygraph dev branch...${NC}"

    if [ -d "$LM_POLYGRAPH_DIR" ]; then
        echo -e "  Pulling latest changes..."
        cd "$LM_POLYGRAPH_DIR"
        git pull origin dev 2>&1 | grep -E "(Already|Updating)" || true
        cd "$SCRIPT_DIR"
    else
        echo -e "  Cloning lm-polygraph dev branch..."
        git clone -b dev https://github.com/IINemo/lm-polygraph.git
    fi

    echo -e "  Installing lm-polygraph..."
    pip install -e "$LM_POLYGRAPH_DIR" > /dev/null
    echo -e "${GREEN}✓ lm-polygraph installed${NC}"
}

if [ "$UPDATE_ONLY" = true ]; then
    install_lm_polygraph
    exit 0
fi

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  LLM TTS Service Setup${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Install package dependencies via pip
echo -e "${YELLOW}Installing package dependencies...${NC}"
pip install -e . > /dev/null
echo -e "${GREEN}✓ Package installed${NC}\n"

# Install lm-polygraph dev branch
install_lm_polygraph

echo -e "\n${GREEN}✅ Setup complete!${NC}"
echo -e "\nNext: Copy .env.example to .env and add your API keys"
echo -e "Update lm-polygraph: ${BLUE}./setup.sh --update${NC}"
