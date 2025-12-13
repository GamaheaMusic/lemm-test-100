# Simple HuggingFace Spaces Deployment Script
param(
    [string]$SpaceDir = ".\lemm-test-100",
    [string]$Token = ""
)

$ErrorActionPreference = "Stop"

Write-Host "HuggingFace Spaces Deployment" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

# Clone if needed
if (-not (Test-Path $SpaceDir)) {
    Write-Host "`nCloning HuggingFace Space..." -ForegroundColor Yellow
    git clone https://huggingface.co/spaces/Gamahea/lemm-test-100 $SpaceDir
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to clone repository" -ForegroundColor Red
        exit 1
    }
    Write-Host "Repository cloned" -ForegroundColor Green
}

Write-Host "`nPreparing files for deployment..." -ForegroundColor Yellow

# Copy files
Copy-Item "app.py" "$SpaceDir/" -Force
Copy-Item "hf_config.py" "$SpaceDir/" -Force
Copy-Item "requirements_hf.txt" "$SpaceDir/requirements.txt" -Force
Copy-Item "packages.txt" "$SpaceDir/" -Force
Copy-Item "README_HF.md" "$SpaceDir/README.md" -Force
Copy-Item "pre_startup.sh" "$SpaceDir/" -Force
Copy-Item "setup_diffrhythm2_src.sh" "$SpaceDir/" -Force

# Copy backend
if (Test-Path "$SpaceDir\backend") {
    Remove-Item "$SpaceDir\backend" -Recurse -Force
}
Copy-Item "backend" "$SpaceDir\" -Recurse -Force -Exclude @('__pycache__','*.pyc','*.log','logs')

# Create .gitignore
@"
__pycache__/
*.pyc
*.pyo
.Python
*.log
/models/
outputs/
logs/
.env
"@ | Out-File "$SpaceDir\.gitignore" -Encoding ASCII

Write-Host "Files prepared" -ForegroundColor Green

# Git operations
Write-Host "`nCommitting and pushing..." -ForegroundColor Yellow

Push-Location $SpaceDir

try {
    # Configure git if token provided
    if ($Token) {
        git config credential.helper store
        "https://user:$Token@huggingface.co" | Out-File ~/.git-credentials -Encoding ASCII
    }
    
    git add .
    git commit -m "Deploy Music Generation Studio - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    git push
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nDeployment complete!" -ForegroundColor Green
        Write-Host "Your Space: https://huggingface.co/spaces/Gamahea/lemm-test-100" -ForegroundColor Cyan
    } else {
        Write-Host "Push failed" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}
