# reset_qdrant_and_create_collection.ps1
# Usage: Open PowerShell in project dir, run: .\reset_qdrant_and_create_collection.ps1

# === CONFIG ===
$StoragePath = "C:\qdrant_storage"
$CollectionName = "asklyne_collection"
$VectorDim = 1024        # BAAI/bge-large-en-v1.5 -> 1024 dims
$QdrantImage = "qdrant/qdrant:latest"
$ContainerName = "qdrant"
# =============

Write-Host "Stopping and removing existing qdrant container (if any)..."
docker stop $ContainerName -t 5 2>$null | Out-Null
docker rm $ContainerName 2>$null | Out-Null

Write-Host "Removing old storage (destructive) -> $StoragePath"
if (Test-Path $StoragePath) {
    Remove-Item -LiteralPath $StoragePath -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path $StoragePath -Force | Out-Null

Write-Host "Starting fresh qdrant container..."
docker run -d --name $ContainerName `
  -p 6333:6333 -p 6334:6334 `
  -v "${StoragePath}:/qdrant/storage" `
  $QdrantImage | Out-Null

Write-Host "Waiting 4 seconds for Qdrant to start..."
Start-Sleep -Seconds 4

Write-Host "Tail last 80 lines of qdrant logs (if more needed, run docker logs -f qdrant)..."
docker logs qdrant --tail 80

# set env var for current PowerShell session (also write user env so next app restart picks it)
Write-Host "Setting EMBED_MODEL_TEXT to BAAI/bge-large-en-v1.5 for current session and for current user."
$env:EMBED_MODEL_TEXT = "BAAI/bge-large-en-v1.5"
[System.Environment]::SetEnvironmentVariable('EMBED_MODEL_TEXT','BAAI/bge-large-en-v1.5','User')

# Create Qdrant collection with correct vectors JSON
$body = @{
  vectors = @{
    default = @{
      size = $VectorDim
      distance = "Cosine"
    }
  }
} | ConvertTo-Json -Depth 10

Write-Host "Creating collection $CollectionName with vector dim $VectorDim..."
try {
    $resp = Invoke-RestMethod -Uri "http://localhost:6333/collections/$CollectionName" -Method PUT -Body $body -ContentType "application/json" -ErrorAction Stop
    Write-Host "Create collection response:" (ConvertTo-Json $resp -Depth 6)
} catch {
    Write-Host "Create collection failed. Full error:"
    Write-Host $_.Exception.Response.Content.ReadAsStringAsync().Result
}

Write-Host "Verify collection:"
Invoke-RestMethod -Uri "http://localhost:6333/collections/$CollectionName" -Method GET | ConvertTo-Json -Depth 6
Write-Host "Done. Now restart your Python app/process so it sees EMBED_MODEL_TEXT environment var."
