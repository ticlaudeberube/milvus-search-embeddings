# Ollama Optimization Scripts

## Cross-Platform Support

### Windows (PowerShell)
- `set-ollama-optimization.ps1`
- Uses `[System.Environment]::SetEnvironmentVariable()` for permanent storage
- Sets variables in Windows user environment

### macOS/Linux (Shell)
- `set-ollama-optimization.sh`
- Adds variables to shell config files (`~/.zshrc`, `~/.bashrc`, `~/.profile`)
- Auto-detects shell type

## What It Does

This shell script sets optimized environment variables for running Ollama inference on your system. It configures key performance settings—like thread count and parallelism—and stores them permanently.

### Environment Variables Set
- `OLLAMA_NUM_THREADS` - matches your CPU's logical thread count
- `OLLAMA_NUM_PARALLEL` - higher for multitasking, lower for single-user  
- `OLLAMA_MAX_LOADED_MODELS` - depends on RAM availability
- `OLLAMA_CUDA` - if using a supported AMD GPU or ROCm

### Features
- Adds variables to `~/.zshrc` (or `~/.bashrc`) to ensure they persist across sessions
- Displays confirmation messages for each variable set
- Automatically detects shell config file

## Why Use It

- Ensures Ollama runs with optimal performance tailored to your CPU
- Saves time by avoiding manual configuration each session  
- Easily updatable: just rerun with new values to overwrite old ones

## Usage

### Windows
```powershell
# Run PowerShell script
.\environments\set-ollama-optimization.ps1
```

### macOS/Linux
```bash
# Make executable and run
chmod +x environments/set-ollama-optimization.sh
./environments/set-ollama-optimization.sh

# Apply changes
source ~/.zshrc  # or ~/.bashrc
```

## Platform-Specific Details

### Windows (PowerShell)
- Variables persist across reboots using Windows user environment
- No restart required for current session
- Uses Windows registry for permanent storage

### macOS/Linux (Shell)
- Auto-detects shell type and uses appropriate config file:
  - **zsh**: `~/.zshrc`
  - **bash**: `~/.bashrc` 
  - **other**: `~/.profile`
- Removes existing entries before adding new ones
- Sets variables for current session immediately