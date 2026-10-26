# Train and evaluate with the default config, or pass another one: scripts\train.ps1 my_config.yaml
param([string]$Config = "config.yaml")
Set-Location (Join-Path $PSScriptRoot "..")
python main.py --config $Config
