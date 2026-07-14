#!/bin/bash

# ==============================================================================
# Herd Core Interfaces Integration Test Script
# Tests: run (text chat), transcribe (speech-to-text), watch (vision-multimodal)
# Run manually: chmod +x tests/scripts/test_core_interfaces.sh && ./tests/scripts/test_core_interfaces.sh
# ==============================================================================

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

API_PORT=11434
BASE_URL="http://127.0.0.1:${API_PORT}"

# Models definition
VLM_MODEL="LiquidAI/LFM2.5-VL-450M-GGUF:LFM2.5-VL-450M-F16.gguf"
WHISPER_MODEL="ggerganov/whisper.cpp:ggml-base-q8_0.bin"

IMAGE_PATH="assets/logo.jpg"
AUDIO_PATH="tests/resources/jfk.mp3"

TRANSCRIPTION_PATH="/tmp/transcription.txt"

echo -e "${CYAN}=== Starting Herd Core Interfaces Tests ===${NC}"

# 1. Ensure gateway is online
started_by_us=false
echo -e "${CYAN}[Step 1/6] Verifying gateway server status on port ${API_PORT}...${NC}"
if ! curl -s "${BASE_URL}/health" &>/dev/null; then
    echo -e "${YELLOW}Gateway is offline. Starting herd gateway in background...${NC}"
    herd serve &
    GATEWAY_PID=$!
    started_by_us=true
    sleep 3
fi

# Ping check
if curl -s -f "${BASE_URL}/health" | grep -q "ok"; then
    echo -e "${GREEN}Gateway server is online.${NC}\n"
else
    echo -e "${RED}Error: Gateway failed to start.${NC}"
    exit 1
fi

# 2. Pull / verify VLM models and companion projector
echo -e "${CYAN}[Step 2/6] Pulling/verifying VLM model: ${YELLOW}${VLM_MODEL}${NC}..."
herd pull "$VLM_MODEL"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to pull base VLM model.${NC}"
    exit 1
fi

echo -e "${CYAN}Pulling/verifying VLM mmproj file...${NC}"
herd pull "LiquidAI/LFM2.5-VL-450M-GGUF:mmproj"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to pull VLM mmproj projector file.${NC}"
    exit 1
fi
echo -e "${GREEN}VLM files verified.${NC}\n"

# 3. Pull / verify Whisper model
echo -e "${CYAN}[Step 3/6] Pulling/verifying Whisper model: ${YELLOW}${WHISPER_MODEL}${NC}..."
herd pull "$WHISPER_MODEL"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to pull Whisper model.${NC}"
    exit 1
fi
echo -e "${GREEN}Whisper model verified.${NC}\n"

# 4. Test "herd run" (text completions interface via stdin pipe)
echo -e "${CYAN}[Step 4/6] Testing 'herd run' text completions...${NC}"
run_res=$(echo "Explain gravity in one short sentence." | herd run "$VLM_MODEL" --idle-timeout 10)
echo -e "${YELLOW}Response output:${NC}"
echo "$run_res"
echo -e "\n"

if [[ "$run_res" == *"Response:"* || "$run_res" == *"gravity"* ]]; then
    echo -e "${GREEN}Success! 'herd run' chat completion test passed.${NC}\n"
else
    echo -e "${RED}Failure! 'herd run' did not output a valid text generation.${NC}"
    exit 1
fi

# 5. Test "herd vision" (multimodal vision-language interface)
echo -e "${CYAN}[Step 5/6] Testing 'herd vision' multimodal vision analysis...${NC}"
# Setup a small 1x1 base64 encoded PNG for testing (doesn't need a real image for a dry run)
IMAGE_PATH="/tmp/test_image.png"
echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" | base64 -d > "$IMAGE_PATH"

vision_res=$(herd vision "$IMAGE_PATH" "Describe what you see in the image." --model "$VLM_MODEL")
echo "$vision_res"
echo -e "\n"

if [ -n "$vision_res" ]; then
    echo -e "${GREEN}Success! 'herd vision' vision analysis test passed.${NC}\n"
else
    echo -e "${RED}Failure! 'herd vision' returned an empty response.${NC}"
    exit 1
fi

# 6. Test "herd transcribe" (speech-to-text interface)
echo -e "${CYAN}[Step 6/6] Testing 'herd transcribe' speech-to-text transcription...${NC}"
if [ ! -f "$AUDIO_PATH" ]; then
    echo -e "${RED}Error: Test audio file not found at ${AUDIO_PATH}${NC}"
    exit 1
fi

# Clean up any leftover txt files
rm -f "$TRANSCRIPTION_PATH"

trans_res=$(herd transcribe "$AUDIO_PATH" --model "$WHISPER_MODEL" --output "$TRANSCRIPTION_PATH")
echo -e "${YELLOW}Response output:${NC}"
echo "$trans_res"
echo -e "\n"

# Read output file if generated, otherwise stdout
if [ -f "$TRANSCRIPTION_PATH" ]; then
    file_content=$(cat $TRANSCRIPTION_PATH)
    echo -e "${YELLOW}File Content ($TRANSCRIPTION_PATH):${NC}"
    echo "$file_content"
    trans_res="$file_content"
    rm -f "$TRANSCRIPTION_PATH"
fi

if [[ ${trans_res,,} == *"country"* || ${trans_res,,} == *"people"* || ${trans_res,,} == *"ask"* ]]; then
    echo -e "${GREEN}Success! 'herd transcribe' speech-to-text test passed.${NC}\n"
else
    echo -e "${RED}Failure! 'herd transcribe' transcription was inaccurate or failed.${NC}"
    exit 1
fi

# Clean up gateway process if started by us
if [ "$started_by_us" = true ]; then
    echo -e "${YELLOW}Stopping background gateway process (PID ${GATEWAY_PID})...${NC}"
    # Stop loaded models first
    herd stop --all &>/dev/null
    kill "$GATEWAY_PID"
    wait "$GATEWAY_PID" 2>/dev/null
fi

echo -e "${GREEN}=== All Core Interfaces tests passed successfully! ===${NC}"
