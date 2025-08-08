
# Detect shell config file
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
else
    SHELL_CONFIG="$HOME/.profile"
fi

# Environment variables to set
declare -A env_vars=(
    ["OLLAMA_NUM_THREADS"]="24"       # matches your CPU's logical thread count
    ["OLLAMA_NUM_PARALLEL"]="4"       # higher for multitasking, lower for single-user
    ["OLLAMA_MAX_LOADED_MODELS"]="2"  # depends on RAM availability
    ["OLLAMA_CUDA"]="1"               # if using a supported AMD GPU or ROCm
)

# Remove existing entries and add new ones
for key in "${!env_vars[@]}"; do
    value="${env_vars[$key]}"
    
    # Remove existing entry if it exists
    sed -i.bak "/^export $key=/d" "$SHELL_CONFIG" 2>/dev/null || true
    
    # Add new entry
    echo "export $key=\"$value\"" >> "$SHELL_CONFIG"
    
    # Set for current session
    export "$key"="$value"
    
    echo "Set $key to $value"
done

echo ""
echo "✅ Ollama environment variables have been set permanently in $SHELL_CONFIG"
echo "Run 'source $SHELL_CONFIG' or restart your terminal for changes to take effect."