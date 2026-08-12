[CmdletBinding()]
param(
    [string]$Version = "v1.9.2",
    [string]$SourceRoot = "C:\Tools\whisper-parakeet-v1.9.2",
    [string]$InstallRoot = "C:\Tools\parakeet",
    [string]$ModelPath = "C:\Models\parakeet\ggml-parakeet-tdt-0.6b-v3-q8_0.bin",
    [switch]$SkipModel
)

$ErrorActionPreference = "Stop"
$expectedHash = "4D64E9E96C2792186D072FDE0034DF0AD670CF680A2F53069052EAD827FD600E"
$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake) { $cmake = "C:\Program Files\CMake\bin\cmake.exe" }
if (-not (Test-Path $cmake)) { throw "CMake was not found. Install Kitware.CMake first." }
if (-not (Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools")) {
    throw "Visual Studio 2022 Build Tools with the C++ workload are required."
}

if (-not (Test-Path $SourceRoot)) {
    git clone --depth 1 --branch $Version https://github.com/ggml-org/whisper.cpp.git $SourceRoot
}

$buildRoot = Join-Path $InstallRoot "build-$($Version.TrimStart('v'))"
& $cmake -S $SourceRoot -B $buildRoot -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=OFF
& $cmake --build $buildRoot --config Release --target parakeet-cli --parallel
$binary = Join-Path $buildRoot "bin\Release\parakeet-cli.exe"
if (-not (Test-Path $binary)) { throw "Parakeet build completed without $binary" }

if (-not $SkipModel) {
    $modelDirectory = Split-Path -Parent $ModelPath
    New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null
    if (-not (Test-Path $ModelPath)) {
        Invoke-WebRequest -Uri "https://huggingface.co/ggml-org/parakeet-GGUF/resolve/main/ggml-parakeet-tdt-0.6b-v3-q8_0.bin" -OutFile $ModelPath
    }
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ModelPath).Hash
    if ($actualHash -ne $expectedHash) { throw "Parakeet model SHA-256 mismatch: $actualHash" }
}

& $binary --help | Out-Null
Write-Host "Parakeet CLI: $binary"
if (-not $SkipModel) { Write-Host "Parakeet model: $ModelPath" }
