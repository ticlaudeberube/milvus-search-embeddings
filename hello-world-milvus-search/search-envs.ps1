# Environment variables for hello-world search demo

$env:MY_DB_NAME = "default"
$env:MY_COLLECTION_NAME = "hello_world_collection"

Write-Host "Environment variables set:"
Write-Host "MY_DB_NAME: $env:MY_DB_NAME"
Write-Host "MY_COLLECTION_NAME: $env:MY_COLLECTION_NAME"