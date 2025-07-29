function Enter-MilvusProject {
    Set-Location "c:\Users\claud\Documents\workspace\milvus-search-embeddings"
    if (Test-Path ".\.venv\Scripts\Activate.ps1") {
        .\.venv\Scripts\Activate.ps1
        Write-Host "Virtual environment activated" -ForegroundColor Green
        python --version
    }
}

Set-Alias -Name "milvus" -Value Enter-MilvusProject