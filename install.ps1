$ErrorActionPreference = "Stop"
$repository = if ($env:ML_RUNTIME_REPOSITORY) { $env:ML_RUNTIME_REPOSITORY } else { "rkendel1/rust-ml-runtime" }
$version = $env:ML_RUNTIME_VERSION
if (-not $version) {
  $latest = Invoke-WebRequest -UseBasicParsing "https://github.com/$repository/releases/latest"
  $version = ($latest.BaseResponse.ResponseUri.AbsoluteUri -split "/v")[-1]
}
$version = $version.TrimStart("v")
if (-not [Environment]::Is64BitOperatingSystem) { throw "Only 64-bit Windows is supported" }
$artifact = "ml-runtime-v$version-windows-x86_64.tar.gz"
$base = if ($env:ML_RUNTIME_RELEASE_BASE) { $env:ML_RUNTIME_RELEASE_BASE } else { "https://github.com/$repository/releases/download/v$version" }
$installDir = if ($env:ML_RUNTIME_INSTALL_DIR) { $env:ML_RUNTIME_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "ml-runtime\bin" }
$temporary = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid())
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
  $archive = Join-Path $temporary $artifact
  $checksum = "$archive.sha256"
  Invoke-WebRequest -UseBasicParsing "$base/$artifact" -OutFile $archive
  Invoke-WebRequest -UseBasicParsing "$base/$artifact.sha256" -OutFile $checksum
  $expected = ((Get-Content $checksum -Raw).Trim() -split "\s+")[0]
  if ($expected -notmatch "^[0-9a-fA-F]{64}$") { throw "Invalid checksum file" }
  $actual = (Get-FileHash -Algorithm SHA256 $archive).Hash
  if ($actual -ne $expected) { throw "Checksum verification failed for $artifact" }
  tar -xzf $archive -C $temporary
  $binary = Join-Path $temporary "ml-runtime-v$version-windows-x86_64\ml-runtime.exe"
  if (-not (Test-Path $binary -PathType Leaf)) { throw "Verified archive does not contain ml-runtime.exe" }
  New-Item -ItemType Directory -Force -Path $installDir | Out-Null
  Copy-Item $binary (Join-Path $installDir "ml-runtime.exe") -Force
  Write-Host "Installed ml-runtime $version to $installDir\ml-runtime.exe"
  if (($env:PATH -split ";") -notcontains $installDir) { Write-Host "Add $installDir to PATH to run ml-runtime from any shell." }
} finally {
  Remove-Item -Recurse -Force $temporary -ErrorAction SilentlyContinue
}
