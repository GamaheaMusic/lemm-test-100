#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Music Generation App - Universal Launcher
.DESCRIPTION
    Handles first-time setup (venv, requirements, models) and launches the application.
    Run this script to start the app - it will handle everything automatically.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function Write-Header {
    Write-Host "`n================================================================" -ForegroundColor Cyan
    Write-Host "   Music Generation App - Launcher" -ForegroundColor Cyan
    Write-Host "================================================================`n" -ForegroundColor Cyan
}

function Write-Success { Write-Host "[OK] $args" -ForegroundColor Green }
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-ErrorMsg { Write-Host "[ERROR] $args" -ForegroundColor Red }
function Write-Step { Write-Host "[...] $args" -ForegroundColor Yellow }

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

function Initialize-Environment {
    Write-Header
    Write-Info "Checking Python environment..."
    
    $venvPath = Join-Path $ProjectRoot ".venv"
    $pythonExe = Join-Path $venvPath "Scripts\python.exe"
    
    # Check if virtual environment exists
    if (-not (Test-Path $pythonExe)) {
        Write-Step "Creating virtual environment..."
        
        # Find Python 3.11
        $python = (Get-Command py -ErrorAction SilentlyContinue)
        if ($python) {
            & py -3.11 -m venv $venvPath
        } else {
            $python = (Get-Command python -ErrorAction SilentlyContinue)
            if ($python) {
                & python -m venv $venvPath
            } else {
                Write-ErrorMsg "Python not found! Please install Python 3.11"
                exit 1
            }
        }
        
        if (Test-Path $pythonExe) {
            Write-Success "Virtual environment created"
        } else {
            Write-ErrorMsg "Failed to create virtual environment"
            exit 1
        }
    } else {
        Write-Success "Virtual environment found"
    }
    
    return $pythonExe
}

function Install-Dependencies {
    param([string]$PythonExe)
    
    Write-Info "Checking dependencies..."
    
    $requirementsFile = Join-Path $ProjectRoot "requirements.txt"
    $flagFile = Join-Path $ProjectRoot ".venv\.deps_installed"
    
    # Check if we need to install/update
    $needsInstall = $false
    if (-not (Test-Path $flagFile)) {
        $needsInstall = $true
    } else {
        $reqTime = (Get-Item $requirementsFile).LastWriteTime
        $flagTime = (Get-Item $flagFile).LastWriteTime
        if ($reqTime -gt $flagTime) {
            $needsInstall = $true
            Write-Step "Requirements updated - reinstalling..."
        }
    }
    
    if ($needsInstall) {
        Write-Step "Installing Python dependencies (this may take a few minutes)..."
        
        # Upgrade pip first
        & $PythonExe -m pip install --upgrade pip --quiet
        
        # Install requirements
        & $PythonExe -m pip install -r $requirementsFile --quiet
        
        if ($LASTEXITCODE -eq 0) {
            # Create flag file
            New-Item -ItemType File -Path $flagFile -Force | Out-Null
            Write-Success "Dependencies installed successfully"
        } else {
            Write-ErrorMsg "Failed to install dependencies"
            exit 1
        }
    } else {
        Write-Success "Dependencies up to date"
    }
    
    # Install DiffRhythm2 additional dependencies
    $dr2FlagFile = Join-Path $ProjectRoot ".venv\.dr2_deps_installed"
    
    if (-not (Test-Path $dr2FlagFile)) {
        Write-Step "Installing DiffRhythm2 dependencies..."
        
        & $PythonExe -m pip install inflect --quiet
        
        New-Item -ItemType File -Path $dr2FlagFile -Force | Out-Null
        Write-Success "DiffRhythm2 dependencies configured"
    }
}

function Initialize-Models {
    param([string]$PythonExe)
    
    Write-Info "Checking AI models..."
    
    $modelsDir = Join-Path $ProjectRoot "models\diffrhythm2"
    $modelCheck = Join-Path $modelsDir "model.safetensors"
    
    if (Test-Path $modelCheck) {
        Write-Success "DiffRhythm2 model found"
        return $true
    }
    
    Write-Step "Models not found - will be downloaded on first generation"
    Write-Info "Models will be automatically downloaded from HuggingFace (~5GB)"
    return $true
}

# ============================================================================
# SERVER MANAGEMENT
# ============================================================================

function Start-Backend {
    param([string]$PythonExe)
    
    Write-Info "Starting backend server..."
    
    # Use wrapper script that sets environment variables
    $wrapperScript = Join-Path $ProjectRoot "backend\start_with_env.py"
    
    # Start backend process in minimized window
    $processArgs = @{
        FilePath = $PythonExe
        ArgumentList = @($wrapperScript)
        WorkingDirectory = $ProjectRoot
        WindowStyle = 'Minimized'
    }
    
    Start-Process @processArgs
    
    # Wait for backend to be ready
    Write-Step "Waiting for backend to start..."
    $maxWait = 45
    $waited = 0
    
    while ($waited -lt $maxWait) {
        Start-Sleep -Seconds 2
        $waited += 2
        
        # Try to connect to health endpoint
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:7860/api/health" -Method GET -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Success "Backend server ready at http://localhost:7860"
                return $true
            }
        } catch {
            # Still waiting...
        }
    }
    
    Write-ErrorMsg "Backend failed to respond within ${maxWait}s"
    
    # One last check
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:7860/api/health" -Method GET -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Success "Backend server ready at http://localhost:7860"
            return $true
        }
    } catch {
        Write-ErrorMsg "Could not connect to backend"
    }
    
    # Kill any Python processes that might be running
    Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)" -ErrorAction SilentlyContinue).CommandLine
        $cmd -like "*backend*run.py*" -or $cmd -like "*start_with_env.py*"
    } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    return $null
}

function Start-Frontend {
    param([string]$PythonExe)
    
    Write-Info "Starting frontend server..."
    
    $frontendDir = Join-Path $ProjectRoot "frontend"
    
    Write-Success "Frontend server ready at http://localhost:8000"
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "   Music Generation App is running!" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "   Open your browser to: http://localhost:8000" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   Backend API:  http://localhost:7860/api" -ForegroundColor Gray
    Write-Host "   Health Check: http://localhost:7860/api/health" -ForegroundColor Gray
    Write-Host ""
    Write-Host "   Press Ctrl+C to stop both servers" -ForegroundColor Yellow
    Write-Host ""
    
    # Start frontend (blocking)
    & $PythonExe -m http.server 8000 --directory $frontendDir
}

# ============================================================================
# CLEANUP
# ============================================================================

function Stop-Servers {
    Write-Host "`n"
    Write-Info "Shutting down servers..."
    
    # Kill any Python processes running backend or frontend
    Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)" -ErrorAction SilentlyContinue).CommandLine
        $cmd -like "*backend*run.py*" -or $cmd -like "*start_with_env.py*" -or $cmd -like "*http.server*8000*"
    } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    Write-Success "Servers stopped"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

try {
    # Step 1: Initialize Python environment
    $pythonExe = Initialize-Environment
    
    # Step 2: Install dependencies
    Install-Dependencies -PythonExe $pythonExe
    
    # Step 3: Check/setup models
    Initialize-Models -PythonExe $pythonExe
    
    # Step 4: Start backend
    $backendProcess = Start-Backend -PythonExe $pythonExe
    if (-not $backendProcess) {
        Write-ErrorMsg "Cannot start application - backend failed"
        exit 1
    }
    
    # Step 5: Start frontend (blocking until Ctrl+C)
    Start-Frontend -PythonExe $pythonExe
    
} catch {
    Write-ErrorMsg "An error occurred: $_"
    Write-Host $_.ScriptStackTrace
    exit 1
} finally {
    # Cleanup on exit
    Stop-Servers
}
