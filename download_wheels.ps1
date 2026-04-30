# Download Linux Python 3.12 wheels for offline install on the Ubuntu server.
# Run from this directory:
#     powershell -ExecutionPolicy Bypass -File .\download_wheels.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== pip version ==="
python -m pip --version

Write-Host "`n=== Wiping old wheels/ and pip cache ==="
Remove-Item -Recurse -Force wheels -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path wheels | Out-Null
python -m pip cache purge | Out-Null

Write-Host "`n=== Downloading wheels (target: Ubuntu Linux x86_64, CPython 3.12) ==="
# Single command on one logical line — no PowerShell backtick continuations
# so it can't be silently broken by an editor or a missing backtick.
python -m pip download --dest wheels --requirement requirements.txt --python-version 3.12 --implementation cp --abi cp312 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 --only-binary=:all:

Write-Host "`n=== Top-up: deps that pip skips on Windows due to env markers ==="
# Some packages declare deps with environment markers that evaluate to
# False on Windows but True on Linux. pip download evaluates markers
# against the LOCAL machine, so it skips them entirely — even with
# --platform manylinux. Two real cases in this project's tree:
#
#   uvicorn[standard] -> uvloop
#       marker: sys_platform != "win32" and ...
#       (uvloop is Linux/macOS only)
#
#   huggingface-hub -> hf-xet
#       marker: platform_machine == "x86_64" or "amd64" or "arm64" or "aarch64"
#       (case-sensitive: Windows reports "AMD64", marker uses "amd64", so False)
#
# Re-pull each one explicitly with --no-deps so it lands in wheels/.
$linuxOnlyDeps = @(
    "uvloop>=0.15.1"
    "hf-xet>=1.1.3,<2.0.0"
    # add more here if other conditional deps surface
)
foreach ($dep in $linuxOnlyDeps) {
    Write-Host "  -> $dep"
    python -m pip download --dest wheels --no-deps --python-version 3.12 --implementation cp --abi cp312 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 --only-binary=:all: $dep
}

Write-Host "`n=== Verifying wheels target the right platform ==="

# A wheel is OK if it is:
#   1. cp312-cp312-manylinux*       (built specifically for CPython 3.12 on Linux)
#   2. cpXX-abi3-manylinux*          (stable ABI — runs on any CPython >= XX)
#   3. py3-none-any / py2.py3-none-any (pure Python — runs anywhere)
#
# Anything else (Windows, macOS, musllinux Alpine, or a wrong cp interpreter
# tag without abi3) means the cross-platform flags did not apply.
$wrong = Get-ChildItem wheels | Where-Object {
    $n = $_.Name
    if ($n -match 'win_amd64|win32|macosx|musllinux') { return $true }
    if ($n -match '-cp312-cp312-') { return $false }
    if ($n -match '-cp\d+-abi3-')  { return $false }   # stable ABI — always OK
    if ($n -match '-py3-none-any') { return $false }
    if ($n -match '-py2\.py3-none-any') { return $false }
    if ($n -match '-cp(\d+)-cp\d+-') { return $matches[1] -ne '312' }
    return $false
}
if ($wrong) {
    Write-Host "FAIL: Wrong-platform wheels found:" -ForegroundColor Red
    $wrong | ForEach-Object { Write-Host "  $($_.Name)" -ForegroundColor Red }
    Write-Host "`nThe download flags did not apply. Check your pip version (need >= 24.0):" -ForegroundColor Yellow
    python -m pip --version
    exit 1
}

$total = (Get-ChildItem wheels).Count
$linux = (Get-ChildItem wheels | Where-Object { $_.Name -match 'manylinux' }).Count
$pure  = (Get-ChildItem wheels | Where-Object { $_.Name -match 'py3-none-any' }).Count

Write-Host "OK: $total wheels total" -ForegroundColor Green
Write-Host "    $linux Linux x86_64 wheels (cp312)"
Write-Host "    $pure pure-Python wheels (py3-none-any)"
Write-Host "    $($total - $linux - $pure) other`n"

Write-Host "Critical packages present:"
foreach ($pkg in @('chromadb', 'sqlalchemy', 'pydantic_core', 'onnxruntime', 'tokenizers', 'numpy', 'uvicorn', 'uvloop', 'hf_xet')) {
    $hit = Get-ChildItem wheels | Where-Object { $_.Name -ilike "${pkg}-*" } | Select-Object -First 1
    if ($hit) {
        Write-Host "  [OK] $($hit.Name)" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $pkg" -ForegroundColor Yellow
    }
}

Write-Host "`nReady to transfer. Suggested:" -ForegroundColor Cyan
Write-Host "  scp -r wheels requirements.txt user@ubuntu-host:~/multiagent_rag/"
