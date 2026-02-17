$ErrorActionPreference='Stop'
Set-ExecutionPolicy -Scope Process Bypass -Force
$wt = 'c:\Users\pedisa94\Documents\Github_projects\ONH_expert_connector_app.worktrees\step06-chunk-budget'
$prompt = Get-Content -Raw -Path (Join-Path $wt 'worker_prompt_step06.txt')
$codex = (Get-Command codex).Source
Set-Location 'c:\Users\pedisa94\Documents\Github_projects\ONH_expert_connector_app'
.\scripts\run_with_timeout.ps1 -TimeoutSec 10800 -WorkingDirectory $wt -FilePath $codex -ArgumentList @('exec','--dangerously-bypass-approvals-and-sandbox','--cd',$wt,'-o',(Join-Path $wt 'reports/worker_step06_last_message.txt'),$prompt)
