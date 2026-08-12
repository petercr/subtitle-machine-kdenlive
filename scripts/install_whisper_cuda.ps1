[CmdletBinding()]
param(
    [string]$InstallRoot = "C:\Tools\whisper-cuda\12.4",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$downloadUrl = "https://github.com/ggml-org/whisper.cpp/releases/download/v1.8.5/whisper-cublas-12.4.0-bin-x64.zip"
$expectedSha256 = "FF50101F85A6026D39053771C25B42F5752AC05D5BE9EE2E5D2632541ADEF231"
$binaryPath = Join-Path $InstallRoot "Release\whisper-cli.exe"
$archivePath = Join-Path ([System.IO.Path]::GetTempPath()) "whisper-cublas-12.4.0-bin-x64.zip"

if ((Test-Path $binaryPath) -and -not $Force) {
    Write-Host "CUDA Whisper is already installed at $binaryPath"
    exit 0
}

try {
    Write-Host "Downloading official whisper.cpp CUDA bundle..."
    Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath

    $actualSha256 = (Get-FileHash $archivePath -Algorithm SHA256).Hash
    if ($actualSha256 -ne $expectedSha256) {
        throw "SHA-256 mismatch. Expected $expectedSha256 but received $actualSha256."
    }

    New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
    Expand-Archive -LiteralPath $archivePath -DestinationPath $InstallRoot -Force
    if (-not (Test-Path $binaryPath)) {
        throw "The archive did not contain the expected executable: $binaryPath"
    }

    & $binaryPath -h *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA Whisper executable failed its startup check with exit code $LASTEXITCODE."
    }
    Write-Host "Installed CUDA Whisper at $binaryPath"
}
finally {
    Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
}
