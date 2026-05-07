# -*- coding: utf-8 -*-
# deploy_a6000.ps1 — 上传当前项目到 A6000；不保存密码
# 用法: powershell -ExecutionPolicy Bypass -File scripts/remote/deploy_a6000.ps1
param(
    [string]$ProjectDir = "d:\vlllm",
    [string]$RemoteHost = "xtraa6k04",
    [string]$RemoteDir = "~/vlllm"
)

$ErrorActionPreference = "Stop"
$archive = Join-Path $env:TEMP "vlllm_a6000_upload.tar.gz"
$parent = Split-Path -Parent $ProjectDir
$leaf = Split-Path -Leaf $ProjectDir

Write-Host "==> 打包项目: $ProjectDir"
if (Test-Path $archive) { Remove-Item $archive -Force }
tar --exclude=.git --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache --exclude=logs --exclude=results -czf $archive -C $parent $leaf

Write-Host "==> 上传到服务器: $RemoteHost"
scp $archive "${RemoteHost}:~/vlllm_upload.tar.gz"

Write-Host "==> 解压到服务器: $RemoteDir"
ssh $RemoteHost "rm -rf $RemoteDir && mkdir -p $RemoteDir && tar -xzf ~/vlllm_upload.tar.gz -C ~ && rm ~/vlllm_upload.tar.gz"

Write-Host "==> 上传完成。下一步："
Write-Host "ssh $RemoteHost 'cd ~/vlllm && bash scripts/remote/bootstrap_a6000.sh && bash scripts/experiment/run_a6000_baseline.sh'"