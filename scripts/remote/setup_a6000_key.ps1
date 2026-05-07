# -*- coding: utf-8 -*-
# setup_a6000_key.ps1 — 为 A6000 配置 SSH 公钥登录；不保存密码
# 用法: powershell -ExecutionPolicy Bypass -File scripts/remote/setup_a6000_key.ps1
param(
    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_vlllm",
    [string]$JumpHost = "nus3090",
    [string]$RemoteHost = "xtraa6k04"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $KeyPath)) {
    Write-Host "==> 生成专用 SSH key: $KeyPath"
    ssh-keygen -t ed25519 -f $KeyPath -N "" -C "vlllm-a6000"
}

$pub = Get-Content "$KeyPath.pub" -Raw
$escaped = $pub.Replace("'", "'\''").Trim()
$remoteCmd = "mkdir -p ~/.ssh; chmod 700 ~/.ssh; touch ~/.ssh/authorized_keys; if ! grep -qxF '$escaped' ~/.ssh/authorized_keys; then echo '$escaped' >> ~/.ssh/authorized_keys; fi; chmod 600 ~/.ssh/authorized_keys"

Write-Host "==> 安装公钥到跳板机: $JumpHost"
Write-Host "    这里会要求你输入一次跳板机密码。"
ssh $JumpHost $remoteCmd

Write-Host "==> 安装公钥到 A6000: $RemoteHost"
Write-Host "    若跳板机公钥已生效，这里只会要求你输入一次 A6000 密码。"
ssh $RemoteHost $remoteCmd

Write-Host "==> 公钥安装完成。之后可直接执行: ssh $RemoteHost"