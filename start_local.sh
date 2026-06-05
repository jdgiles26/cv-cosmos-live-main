#!/bin/bash

# start_local.sh - Local Developer Workspace Launcher for Cosmos Live
# Ensures all local AI models, database services, and media pipelines start smoothly.

set -e

# Setup colors for premium logging
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0;37m' # No Color

echo -e "${CYAN}===================================================================${NC}"
echo -e "${CYAN}             COSMOS LIVE - LOCAL WORKSPACE INITIALIZER             ${NC}"
echo -e "${CYAN}===================================================================${NC}"

# 1. Check if Ollama is running
echo -e "\n${BLUE}[1/5] Checking Ollama Service...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo -e "${GREEN}✔ Ollama is running locally!${NC}"
else
    echo -e "${RED}✘ Ollama is not running. Please start the Ollama application first.${NC}"
    exit 1
fi

# 2. Check Ollama Models
echo -e "\n${BLUE}[2/5] Checking Ollama Models...${NC}"
MODELS_LIST=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null || curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -d':' -f2 | tr -d '"')

if echo "$MODELS_LIST" | grep -q "llama3.1"; then
    echo -e "${GREEN}✔ Found llama3.1:latest for Retrieval LLM Brain${NC}"
else
    echo -e "${YELLOW}⚠ llama3.1 not detected. Attempting to pull...${NC}"
    ollama pull llama3.1:latest
fi

if echo "$MODELS_LIST" | grep -q "llava"; then
    echo -e "${GREEN}✔ Found llava:latest for Ingestion Vision Processing${NC}"
elif echo "$MODELS_LIST" | grep -q "gemma3:4b"; then
    echo -e "${GREEN}✔ Found gemma3:4b for Ingestion Vision Processing${NC}"
else
    echo -e "${YELLOW}⚠ Vision model (llava:latest) not detected. Attempting to pull...${NC}"
    ollama pull llava:latest
fi

# 3. Start Qdrant Docker
echo -e "\n${BLUE}[3/5] Launching local Qdrant Vector Database...${NC}"
if ! docker ps > /dev/null 2>&1; then
    echo -e "${RED}✘ Docker daemon is not running. Please start Docker Desktop first.${NC}"
    exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q '^qdrant$'; then
    echo -e "${YELLOW}Found existing qdrant container. Starting it...${NC}"
    docker start qdrant
else
    echo -e "${YELLOW}Creating and starting new Qdrant container...${NC}"
    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
fi
echo -e "${GREEN}✔ Qdrant is running at http://localhost:6333${NC}"

# 4. Start LiveKit Server
echo -e "\n${BLUE}[4/5] Starting LiveKit Server in Dev Mode...${NC}"
if pgrep livekit-server >/dev/null; then
    echo -e "${YELLOW}Stopping existing livekit-server process...${NC}"
    pkill livekit-server || true
    sleep 1
fi

# Start livekit-server in dev mode in background
livekit-server --dev --bind 0.0.0.0 > livekit-server.log 2>&1 &
LIVEKIT_PID=$!
echo -e "${GREEN}✔ LiveKit server started in background (PID: $LIVEKIT_PID). Logs at livekit-server.log${NC}"
echo -e "${GREEN}✔ URL: ws://localhost:7880, Key: devkey, Secret: secret${NC}"

# 5. Build/Install Local Packages
echo -e "\n${BLUE}[5/5] Checking python virtual environment and project setup...${NC}"
uv sync
echo -e "${GREEN}✔ Workspace dependencies verified and synced successfully!${NC}"

echo -e "\n${CYAN}===================================================================${NC}"
echo -e "${GREEN}🎉 ALL LOCAL SERVICES ARE ACTIVE AND RUNNING 100% OFFLINE!${NC}"
echo -e "${CYAN}===================================================================${NC}"
echo -e "\nYou can now launch the application components in separate terminal windows:"
echo -e "\n${YELLOW}1. Start the Ingestion App (Frames processing):${NC}"
echo -e "   uv run python -m cosmos_live.ingestion_app"
echo -e "\n${YELLOW}2. Start the Retrieval App (Voice Agent):${NC}"
echo -e "   uv run python -m cosmos_live.retrieval_app dev"
echo -e "\n${YELLOW}3. Start the Dummy Publisher (Synthetic camera stream):${NC}"
echo -e "   uv run python -m cosmos_live.publisher_app --identity cam1 --camera-index dummy"
echo -e "\n${YELLOW}4. View the AI Director Stream Live:${NC}"
echo -e "   Open http://localhost:9090 in your browser!"
echo -e "\n${CYAN}===================================================================${NC}"
echo -e "Press Ctrl+C to terminate the background LiveKit server and exit."

# Keep running to hold the LiveKit server process
trap "echo -e '\nStopping LiveKit Server...'; kill $LIVEKIT_PID || true; exit" INT TERM
wait $LIVEKIT_PID
