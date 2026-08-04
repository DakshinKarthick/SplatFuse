$ErrorActionPreference = 'Stop'
$viewerRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'viewer'
$scene = Join-Path $viewerRoot 'public\scenes\mypic1.ply'
if (-not (Test-Path $scene)) {
  throw 'mypic1.ply is missing. Restore SplatFuse/viewer/public/scenes from the portfolio demo data zip.'
}
Push-Location $viewerRoot
try {
  if (-not (Test-Path 'node_modules')) {
    & npm install
    if ($LASTEXITCODE -ne 0) { throw 'npm install failed.' }
  }
  & npm run dev
} finally { Pop-Location }
