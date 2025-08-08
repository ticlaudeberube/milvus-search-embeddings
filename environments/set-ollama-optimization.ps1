
$envVars = @{
    "OLLAMA_NUM_THREADS"       = "24" #(matches your CPU's logical thread count)
    "OLLAMA_NUM_PARALLEL"      = "4" #(higher for multitasking, lower for single-user)
    "OLLAMA_MAX_LOADED_MODELS" = "2" #(depends on RAM availability)
    "OLLAMA_CUDA"              = "1" # (if using a supported AMD GPU or ROCm)
}

foreach ($key in $envVars.Keys) {
    [System.Environment]::SetEnvironmentVariable($key, $envVars[$key], "User")
    Write-Host "Set $key to $($envVars[$key])"
}

Write-Host "`n✅ Ollama environment variables have been set permanently for your user account."
Write-Host "You may need to restart your terminal or system for changes to take full effect."
