# HuggingFace Spaces Deployment Script
# Automates the deployment process

param(
    [string]$SpaceDir = ".\lemm-test-100",
    [switch]$Clone,
    [switch]$Push
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 HuggingFace Spaces Deployment Helper" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Clone the space if requested
if ($Clone) {
    Write-Host "`n📥 Cloning HuggingFace Space..." -ForegroundColor Yellow
    
    if (Test-Path $SpaceDir) {
        Write-Host "❌ Directory already exists: $SpaceDir" -ForegroundColor Red
        Write-Host "Remove it first or use a different path" -ForegroundColor Red
        exit 1
    }
    
    git clone https://huggingface.co/spaces/Gamahea/lemm-test-100 $SpaceDir
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to clone repository" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ Repository cloned" -ForegroundColor Green
}

# Verify space directory exists
if (-not (Test-Path $SpaceDir)) {
    Write-Host "❌ Space directory not found: $SpaceDir" -ForegroundColor Red
    Write-Host "Run with -Clone to clone the repository first" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n📦 Preparing files for deployment..." -ForegroundColor Yellow

# Copy core application files
Write-Host "  Copying app.py..." -ForegroundColor Gray
Copy-Item "app.py" "$SpaceDir/" -Force

Write-Host "  Copying hf_config.py..." -ForegroundColor Gray
Copy-Item "hf_config.py" "$SpaceDir/" -Force

Write-Host "  Copying requirements.txt..." -ForegroundColor Gray
Copy-Item "requirements_hf.txt" "$SpaceDir/requirements.txt" -Force

Write-Host "  Copying packages.txt..." -ForegroundColor Gray
Copy-Item "packages.txt" "$SpaceDir/" -Force

Write-Host "  Copying README.md..." -ForegroundColor Gray
Copy-Item "README_HF.md" "$SpaceDir/README.md" -Force

Write-Host "  Copying pre_startup.sh..." -ForegroundColor Gray
Copy-Item "pre_startup.sh" "$SpaceDir/" -Force

Write-Host "  Copying setup_diffrhythm2_src.sh..." -ForegroundColor Gray
Copy-Item "setup_diffrhythm2_src.sh" "$SpaceDir/" -Force

# Copy backend directory
Write-Host "  Copying backend/..." -ForegroundColor Gray
if (Test-Path "$SpaceDir\backend") {
    Remove-Item "$SpaceDir\backend" -Recurse -Force
}
Copy-Item "backend" "$SpaceDir\" -Recurse -Force -Exclude @('__pycache__','*.pyc','*.log','logs')

# Create .gitignore
Write-Host "  Creating .gitignore..." -ForegroundColor Gray
@"
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.log
*.swp
*.swo
*~
models/
outputs/
logs/
.env
.DS_Store
"@ | Out-File "$SpaceDir\.gitignore" -Encoding ASCII

Write-Host "✅ Files prepared" -ForegroundColor Green

# Show file summary
Write-Host "`n📋 Deployment Summary:" -ForegroundColor Cyan
Write-Host "  Location: $SpaceDir" -ForegroundColor Gray
Write-Host "  Files copied:" -ForegroundColor Gray
Get-ChildItem $SpaceDir -File | ForEach-Object { Write-Host "    - $($_.Name)" -ForegroundColor DarkGray }
Write-Host "  Directories:" -ForegroundColor Gray
Get-ChildItem $SpaceDir -Directory | ForEach-Object { Write-Host "    - $($_.Name)/" -ForegroundColor DarkGray }

# Git operations
if ($Push) {
    Write-Host "`n📤 Committing and pushing to HuggingFace..." -ForegroundColor Yellow
    
    Push-Location $SpaceDir
    
    try {
        Write-Host "  Adding files to git..." -ForegroundColor Gray
        git add .
        
        Write-Host "  Committing changes..." -ForegroundColor Gray
        git commit -m "Deploy Music Generation Studio - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Pushing to HuggingFace..." -ForegroundColor Gray
            git push
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[SUCCESS] Deployment complete!" -ForegroundColor Green
                Write-Host "`nYour Space: https://huggingface.co/spaces/Gamahea/lemm-test-100" -ForegroundColor Cyan
                Write-Host "Build will start automatically. Check the Space page for status." -ForegroundColor Yellow
            } else {
                Write-Host "❌ Failed to push" -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "No changes to commit" -ForegroundColor Gray
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "  1. Review the files in: $SpaceDir" -ForegroundColor Gray
    Write-Host "  2. Run: cd $SpaceDir" -ForegroundColor Gray
    Write-Host "  3. Run: git add ." -ForegroundColor Gray
    Write-Host "  4. Run: git commit -m 'Deploy Music Generation Studio'" -ForegroundColor Gray
    Write-Host "  5. Run: git push" -ForegroundColor Gray
    Write-Host "`n  Or run this script with -Push to automate steps 3-5" -ForegroundColor Cyan
}

Write-Host "`nDeployment preparation complete!" -ForegroundColor Green
