param(
  [Parameter(Mandatory = $true)]
  [string]$FilePath,

  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ArgumentList = @(),

  [int]$TimeoutSec = 900,

  [string]$WorkingDirectory = (Get-Location).Path
)

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $FilePath
$startInfo.WorkingDirectory = $WorkingDirectory
$startInfo.UseShellExecute = $false
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.CreateNoWindow = $true

function _Quote-ProcessArg {
  param([string]$Value)
  if ($null -eq $Value) { return '""' }
  if ($Value -eq '') { return '""' }
  if ($Value -notmatch '[\s"]') { return $Value }
  $escaped = $Value -replace '"', '\\"'
  return '"' + $escaped + '"'
}

$startInfo.Arguments = ($ArgumentList | ForEach-Object { _Quote-ProcessArg $_ }) -join ' '

$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $startInfo

if (-not $process.Start()) {
  throw "Failed to start process: $FilePath"
}

$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()

$timeoutMs = [Math]::Max(1, $TimeoutSec) * 1000
if (-not $process.WaitForExit($timeoutMs)) {
  try {
    & taskkill.exe /PID $process.Id /T /F | Out-Null
  } catch {
    try { $process.Kill($true) } catch {}
  }
  throw "Timed out after ${TimeoutSec}s: $FilePath $($ArgumentList -join ' ')"
}

$process.WaitForExit()
$stdout = $stdoutTask.Result
$stderr = $stderrTask.Result

if ($stdout) { Write-Output $stdout.TrimEnd() }
if ($stderr) { Write-Output $stderr.TrimEnd() }

if ($process.ExitCode -ne 0) {
  throw "Exit code $($process.ExitCode): $FilePath $($ArgumentList -join ' ')"
}
