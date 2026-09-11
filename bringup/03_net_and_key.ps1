$ErrorActionPreference='Continue'
$log='C:\Users\daaddison\jetson-probe\prep.log'
$sp=New-Object System.IO.Ports.SerialPort('COM4',115200,[IO.Ports.Parity]::None,8,[IO.Ports.StopBits]::One)
$sp.Handshake=[IO.Ports.Handshake]::None
$sp.ReadTimeout=300; $sp.DtrEnable=$true; $sp.RtsEnable=$true
$ok=$false
for($t=1;$t -le 25;$t++){ try{ $sp.Open(); $ok=$true; break }catch{ Start-Sleep -Milliseconds 400 } }
if(-not $ok){ Write-Output "COM4 busy"; exit 1 }
function ReadSer { try{ $d=$sp.ReadExisting(); if($d){ Add-Content -Path $log -Value $d -NoNewline; return $d } }catch{}; return '' }
function Settle([int]$ms){ $sb=New-Object Text.StringBuilder; $last=Get-Date; $st=Get-Date
  while(((Get-Date)-$last).TotalMilliseconds -lt $ms){ $d=ReadSer; if($d){[void]$sb.Append($d); $last=Get-Date}; Start-Sleep -Milliseconds 90
    if(((Get-Date)-$st).TotalSeconds -gt 45){break} }
  return ($sb.ToString() -replace "\x1b\[[0-9;?]*[A-Za-z]",'' -replace "\x1b",'') }
function Run([string]$c,[int]$w=6000){
  $sp.Write($c + "`r`n")
  $out = Settle $w
  Write-Host ("`$ " + $c)
  $clean = ($out -split "`n" | Where-Object { $_ -match '\S' -and $_ -notmatch [regex]::Escape($c) -and $_ -notmatch '^\[sudo\]' }) -join "`n"
  if($clean){ Write-Host $clean }
  Write-Host ""
  return $out
}

# clear any half-typed line, then wake the console
$sp.Write([string][char]3); Start-Sleep -Milliseconds 600
$sp.Write("`r`n"); Start-Sleep -Milliseconds 800
$sp.Write("`r`n"); Start-Sleep -Milliseconds 800
$sp.Write("`r`n"); Start-Sleep -Milliseconds 1500
$probe = Settle 5000
Write-Output "--- console probe ---"
Write-Output (($probe -split "`n" | Where-Object { $_ -match '\S' } | Select-Object -Last 6) -join "`n")
if($probe -match 'login:'){
  Write-Output "(login prompt seen - authenticating)"
  $sp.Write("jetson`r`n"); Start-Sleep -Milliseconds 2000; [void](Settle 2000)
  $sp.Write("jetson`r`n"); Start-Sleep -Milliseconds 3000; [void](Settle 5000)
}
$chk = (Run 'echo READY_$((7*7))' 6000)
if($chk -notmatch 'READY_49'){ Write-Output "NO SHELL - aborting"; $sp.Close(); exit 1 }
Write-Output "=== shell confirmed ==="

[void](Run 'mkdir -p ~/.ssh && chmod 700 ~/.ssh' 6000)
[void](Run 'echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJbSs/nCQW5s5KWSCNXKIczpG5FzLcM9VX87WU7JXBie jettank-auto" >> ~/.ssh/authorized_keys' 6000)
[void](Run 'chmod 600 ~/.ssh/authorized_keys; wc -l ~/.ssh/authorized_keys' 6000)
[void](Run 'ip -4 -br addr show enP8p1s0' 6000)
