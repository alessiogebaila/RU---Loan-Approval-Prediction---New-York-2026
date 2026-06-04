# Copy project Kaggle token to the user profile path the CLI reads by default.
$ErrorActionPreference = "Stop"
$projectKaggle = (Resolve-Path (Join-Path $PSScriptRoot "..\..\.kaggle")).Path
$tokenFile = Join-Path $projectKaggle "access_token"
if (-not (Test-Path $tokenFile)) {
    $tokenFile = Join-Path $projectKaggle "kaggle.json"
}
if (-not (Test-Path $tokenFile)) {
    throw "No token file at .kaggle/access_token or .kaggle/kaggle.json"
}

$userKaggle = Join-Path $env:USERPROFILE ".kaggle"
New-Item -ItemType Directory -Force -Path $userKaggle | Out-Null
Copy-Item -Path $tokenFile -Destination (Join-Path $userKaggle "access_token") -Force

$dest = Join-Path $userKaggle "access_token"
icacls $dest /inheritance:r /grant:r "${env:USERNAME}:(R)" | Out-Null
Write-Host "Installed token to $dest"
Write-Host "Verify: kaggle competitions list"
