# Setup, Compilation, & Persistence Guide

This guide details the prerequisite setup, manual and automated binary compilation, and how to run Herd persistently as a system daemon.

---

## 1. System Prerequisites

To use `herd setup` to compile `llama.cpp` and `whisper.cpp` locally, your system must have Git and CMake installed.

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y build-essential cmake git
```

### macOS
Install Xcode Command Line Tools and Homebrew:
```bash
xcode-select --install
brew install cmake git
```

### Windows
1. Install [Git for Windows](https://git-scm.com/download/win).
2. Install [CMake](https://cmake.org/download/).
3. Install Visual Studio Build Tools (make sure to check "Desktop development with C++").

---

## 2. Compilation and Hardware Acceleration

When running `herd setup`, the installer will pull and build the backend libraries. You can accelerate compilation by matching your hardware:

### CPU Compilation (Default)
Compiles utilizing all available logical CPU cores for parallel speed:
```bash
herd setup
```

### CUDA (NVIDIA GPU Acceleration)
Enabling GPU acceleration dramatically speeds up LLM inference by offloading layers to the GPU VRAM. Ensure you have the CUDA Toolkit installed:
```bash
herd setup --cuda
```

### Manual Binary Path Configuration
If you already have precompiled binaries, you can bypass the compilation step. Simply define your binary paths in environment variables or write them directly to your local configuration:

Create `~/.herd/config.json`:
```json
{
  "LLAMA_SERVER_BIN": "/absolute/path/to/your/llama-server",
  "WHISPER_SERVER_BIN": "/absolute/path/to/your/whisper-server"
}
```

---

## 3. Persistent Daemon (Systemd Service Setup)

Running Herd as a systemd user service allows the API gateway to boot up automatically when your system starts, manage its logs under journald, and recover from failures.

### Step 1: Create the User Service Directory
```bash
mkdir -p ~/.config/systemd/user/
```

### Step 2: Create the Service File
Create a new file at `~/.config/systemd/user/herd.service` with the following content:

```ini
[Unit]
Description=Herd Local AI Gateway
After=network.target

[Service]
Type=simple
# Ensure you provide the absolute path to the herd binary in your virtual env
ExecStart=%h/.local/bin/herd serve --host 0.0.0.0
Restart=always
RestartSec=5
# Optional: Set custom environment variables
# Environment=HERD_IDLE_TIMEOUT=600

[Install]
WantedBy=default.target
```
*(Replace `%h/.local/bin/herd` with the actual absolute path to your virtual environment's `herd` executable, which you can find by running `which herd`).*

### Step 3: Enable and Start the Service
Run the following systemd commands to load, enable, and boot the daemon:

```bash
# Reload user daemon configuration
systemctl --user daemon-reload

# Enable the service to run on boot
systemctl --user enable herd.service

# Start the service immediately
systemctl --user start herd.service
```

### Step 4: Verify Service Status
To check if the service is running successfully and inspect logs:

```bash
# Check status
systemctl --user status herd.service

# View live log output
journalctl --user -u herd.service -f
```
