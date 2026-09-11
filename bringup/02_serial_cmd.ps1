# Drive the live initramfs shell on COM4. Commands come from shcmds.txt.
$ErrorActionPreference='Continue'
$log='C:\Users\daaddison\jetson-probe\shell.log'
$sp=New-Object System.IO.Ports.SerialPort('COM4',115200,[IO.Ports.Parity]::None,8,[IO.Ports.StopBits]::One)
$sp.Handshake=[IO.Ports.Handshake]::None
$sp.ReadTimeout=300; $sp.DtrEnable=$true; $sp.RtsEnable=$true
$ok=$false
for($t=1;$t -le 25;$t++){ try{ $sp.Open(); $ok=$true; break }catch{ Start-Sleep -Milliseconds 400 } }
if(-not $ok){ Write-Output "COM4 busy"; exit 1 }
function ReadSer { try{ $d=$sp.ReadExisting(); if($d){ Add-Content -Path $log -Value $d -NoNewline; return $d } }catch{}; return '' }
function Settle([int]$ms){ $sb=New-Object Text.StringBuilder; $last=Get-Date; $st=Get-Date
  while(((Get-Date)-$last).TotalMilliseconds -lt $ms){ $d=ReadSer; if($d){[void]$sb.Append($d); $last=Get-Date}; Start-Sleep -Milliseconds 90
    if(((Get-Date)-$st).TotalSeconds -gt 25){break} }
  return ($sb.ToString() -replace "\x1b\[[0-9;?]*[A-Za-z]",'' -replace "\x1b",'') }

$sp.Write("`r`n"); [void](Settle 1200)
foreach($line in (Get-Content 'C:\Users\daaddison\jetson-probe\shcmds.txt' | Where-Object { $_.Trim().Length -gt 0 })){
  $sp.Write($line + "`r`n")
  $out = Settle 2500
  Write-Output ("`$ " + $line)
  $clean = ($out -split "`n" | Where-Object { $_ -match '\S' -and $_ -notmatch [regex]::Escape($line) }) -join "`n"
  if($clean){ Write-Output $clean }
  Write-Output ""
}
$sp.Close()
