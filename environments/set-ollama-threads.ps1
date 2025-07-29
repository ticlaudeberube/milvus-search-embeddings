# PowerShell version - Ollama thread configuration for optimal performance
$env:OLLAMA_NUM_THREADS = "8"

Write-Host "Ollama threads set to $env:OLLAMA_NUM_THREADS" -ForegroundColor Green