# AURUM-AI V5: Create a Desktop shortcut for the existing .venv desktop launcher.
# No machine-wide changes, no Windows registry writes, no administrator needed.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$script = Join-Path $root 'desktop_launcher.py'
$icon = Join-Path $root 'assets\branding\aurum_ai.ico'
$python = Join-Path $root '.venv\Scripts\pythonw.exe'
if (-not (Test-Path $python)) { $python = Join-Path $root '.venv\Scripts\python.exe' }
foreach ($file in @($script, $icon, $python)) {
    if (-not (Test-Path $file)) { throw "Fichier requis absent : $file" }
}
$desktop = [Environment]::GetFolderPath('DesktopDirectory')
$linkPath = Join-Path $desktop 'AURUM-AI V5.lnk'
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($linkPath)
$link.TargetPath = $python
$link.Arguments = ('"' + $script + '"')
$link.WorkingDirectory = $root
$link.IconLocation = $icon + ',0'
$link.Description = 'AURUM-AI V5 - Historical Investment Simulation (Local Desktop)'
$link.Save()
Write-Host "Raccourci cree : $linkPath"
Write-Host 'Premier lancement : verifier la connexion, les pages et la fermeture de la fenetre.'
