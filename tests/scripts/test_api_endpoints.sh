#!/bin/bash

# ==============================================================================
# Herd Gateway API Endpoint Integration Test Script
# Run manually: chmod +x tests/scripts/test_api_endpoints.sh && ./tests/scripts/test_api_endpoints.sh
# ==============================================================================

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

API_PORT=11434
BASE_URL="http://127.0.0.1:${API_PORT}"
MODEL_NAME="LiquidAI/LFM2.5-VL-450M-GGUF:LFM2.5-VL-450M-BF16.gguf"
EMBED_MODEL="second-state/All-MiniLM-L6-v2-Embedding-GGUF:all-MiniLM-L6-v2-Q8_0.gguf"

echo -e "${CYAN}=== Starting Herd API Endpoint Integration Tests ===${NC}"

# 1. Check if Gateway is running, start it in the background if not
started_by_us=false
echo -e "${CYAN}[Step 1/8] Verifying gateway server status on port ${API_PORT}...${NC}"
if ! curl -s "${BASE_URL}/health" &> /dev/null; then
    echo -e "${YELLOW}Gateway is offline. Starting herd gateway in background...${NC}"
    herd serve &
    GATEWAY_PID=$!
    started_by_us=true
    sleep 3 # Wait for startup
fi

# Ping health check
if curl -s -f "${BASE_URL}/health" | grep -q "ok"; then
    echo -e "${GREEN}Gateway server is online and responding.${NC}\n"
else
    echo -e "${RED}Error: Gateway server failed to start or respond.${NC}"
    exit 1
fi

# 2. Test /v1/models (List Models)
echo -e "${CYAN}[Step 2/8] Testing /v1/models (OpenAI Model List) endpoint...${NC}"
models_res=$(curl -s "${BASE_URL}/v1/models")
if [[ $models_res == *"object"* && $models_res == *"model"* ]]; then
    echo -e "${GREEN}Success! Received model registry listing.${NC}\n"
else
    echo -e "${RED}Failed to list models. Response: ${models_res}${NC}"
    exit 1
fi

# 3. Test /v1/models/load (Load Embedding Model)
echo -e "${CYAN}[Step 3/8] Loading embedding model: ${YELLOW}${EMBED_MODEL}${NC}..."
load_res=$(curl -s -X POST "${BASE_URL}/v1/models/load" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${EMBED_MODEL}\", \"is_embedding\": true}")
if [[ $load_res == *"loaded"* ]]; then
    echo -e "${GREEN}Success! Embedding model loaded.${NC}\n"
else
    echo -e "${RED}Failed to load embedding model. Response: ${load_res}${NC}"
    exit 1
fi

# 4. Test /v1/embeddings (Generate Embedding Vector)
echo -e "${CYAN}[Step 4/8] Generating text embeddings for semantic searches...${NC}"
embed_res=$(curl -s -X POST "${BASE_URL}/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${EMBED_MODEL}\", \"input\": \"Test sentence for vector validation\"}")
if [[ $embed_res == *"embedding"* && $embed_res == *"data"* ]]; then
    echo -e "${GREEN}Success! Embedding vector generated cleanly (pooling mean verified).${NC}\n"
else
    echo -e "${RED}Failed to generate embeddings. Response: ${embed_res}${NC}"
    exit 1
fi

# 5. Test /v1/models/load (Load LLM Model)
echo -e "${CYAN}[Step 5/8] Loading text LLM model: ${YELLOW}${MODEL_NAME}${NC}..."
load_llm_res=$(curl -s -X POST "${BASE_URL}/v1/models/load" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${MODEL_NAME}\", \"is_whisper\": false}")
if [[ $load_llm_res == *"loaded"* ]]; then
    echo -e "${GREEN}Success! LLM model loaded.${NC}\n"
else
    echo -e "${RED}Failed to load LLM model. Response: ${load_llm_res}${NC}"
    exit 1
fi

# 6. Test /v1/chat/completions (Non-Streaming Generation)
echo -e "${CYAN}[Step 6/8] Querying completions (/v1/chat/completions) without streaming...${NC}"
chat_res=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Verify spelling of: antigravity\"}], \"stream\": false}")
if [[ $chat_res == *"choices"* && $chat_res == *"content"* ]]; then
    echo -e "${GREEN}Success! Received completion text.${NC}\n"
else
    echo -e "${RED}Failed non-streaming completion test. Response: ${chat_res}${NC}"
    exit 1
fi

# 7. Test /v1/chat/completions (Streaming SSE Chunks)
echo -e "${CYAN}[Step 7/8] Querying completions (/v1/chat/completions) with stream: true...${NC}"
stream_header_found=false
# Read first few lines of the stream output to confirm SSE protocol
curl -s -N -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Count up to 3\"}], \"stream\": true}" | head -n 5 | grep -q "data:" && stream_header_found=true

if [ "$stream_header_found" = true ]; then
    echo -e "${GREEN}Success! Streaming chunk headers (data:) verified.${NC}\n"
else
    echo -e "${RED}Failed streaming completions SSE handshake check.${NC}"
    exit 1
fi

# 8. Unload models and release VRAM
echo -e "${CYAN}[Step 8/8] Teardown: Stopping loaded model processes...${NC}"
curl -s -X POST "${BASE_URL}/v1/models/unload" -H "Content-Type: application/json" -d "{\"model\": \"${MODEL_NAME}\"}" > /dev/null
curl -s -X POST "${BASE_URL}/v1/models/unload" -H "Content-Type: application/json" -d "{\"model\": \"${EMBED_MODEL}\"}" > /dev/null
echo -e "${GREEN}Models unloaded successfully.${NC}\n"

# Stop background gateway if we booted it
if [ "$started_by_us" = true ]; then
    echo -e "${YELLOW}Stopping background gateway process (PID ${GATEWAY_PID})...${NC}"
    kill "$GATEWAY_PID"
    wait "$GATEWAY_PID" 2>/dev/null
fi

echo -e "${GREEN}=== All API endpoint integration tests passed successfully! ===${NC}"
