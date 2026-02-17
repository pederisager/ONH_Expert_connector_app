param(
  [int]$Port = 8110,
  [string]$OutputPath = 'reports/benchmark_results_100_step06_step08_integrated.json',
  [string]$PythonPath = 'C:\Users\pedisa94\Documents\Github_projects\ONH_expert_connector_app\.venv\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'

$serverArgs = @(
  '-NoProfile',
  '-ExecutionPolicy',
  'Bypass',
  '-Command',
  "Set-ExecutionPolicy -Scope Process Bypass -Force; .\\scripts\\run_with_timeout.ps1 -TimeoutSec 2400 -WorkingDirectory . -FilePath '$PythonPath' -ArgumentList @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port','$Port')"
)

$serverProc = Start-Process -FilePath 'powershell.exe' -ArgumentList $serverArgs -PassThru
$baseUrl = "http://127.0.0.1:$Port"
$ready = $false

try {
  for ($i = 0; $i -lt 180; $i++) {
    try {
      $null = Invoke-WebRequest -Uri "$baseUrl/queue" -UseBasicParsing -TimeoutSec 2
      $ready = $true
      break
    } catch {
      Start-Sleep -Seconds 1
    }
  }

  if (-not $ready) {
    throw "Server did not become ready at $baseUrl within 180 seconds."
  }

  $benchmarkFailed = $false
  try {
    Set-ExecutionPolicy -Scope Process Bypass -Force
    .\scripts\run_with_timeout.ps1 -TimeoutSec 2400 -WorkingDirectory . -FilePath $PythonPath -ArgumentList @('scripts/run_search_benchmark.py','--benchmark','tests/benchmarks/search_relevance_100_v1.yaml','--base-url',$baseUrl,'--output',$OutputPath)
  } catch {
    $benchmarkFailed = $true
    Write-Output "BENCHMARK_EXIT_NONZERO: $($_.Exception.Message)"
  }

  if ($benchmarkFailed) {
    Write-Output 'BENCHMARK_COMPLETED_WITH_THRESHOLD_FAILURES'
  }
} finally {
  if ($serverProc -and -not $serverProc.HasExited) {
    & C:\Windows\System32\taskkill.exe /PID $serverProc.Id /T /F | Out-Null
  }

  Start-Sleep -Seconds 1
  try {
    $null = Invoke-WebRequest -Uri "$baseUrl/queue" -UseBasicParsing -TimeoutSec 2
    Write-Output 'SERVER_STILL_UP'
  } catch {
    Write-Output 'SERVER_DOWN'
  }
}
