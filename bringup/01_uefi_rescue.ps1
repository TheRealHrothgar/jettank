# Catch UEFI -> Boot Manager -> UEFI Shell -> boot to a root shell -> fix OOBE -> reboot.
$ErrorActionPreference='Continue'
$ESC=[string][char]27
$log='C:\Users\daaddison\jetson-probe\rescue.log'
Remove-Item $log -ErrorAction SilentlyContinue
$sp=New-Object System.IO.Ports.SerialPort('COM4',115200,[IO.Ports.Parity]::None,8,[IO.Ports.StopBits]::One)
$sp.Handshake=[IO.Ports.Handshake]::None
$sp.ReadTimeout=200; $sp.DtrEnable=$true; $sp.RtsEnable=$true
$ok=$false
for($t=1;$t -le 30;$t++){ try{ $sp.Open(); $ok=$true; break }catch{ Start-Sleep -Milliseconds 400 } }
if(-not $ok){ Write-Output "COM4 busy"; exit 1 }

function ReadSer {
  try{ $d=$sp.ReadExisting()
       if($d){ Add-Content -Path $log -Value $d -NoNewline
               return ($d -replace "\x1b\[[0-9;?]*[A-Za-z]",'' -replace "\x1b",'') } }catch{}
  return ''
}
function Settle([int]$ms){
  $sb=New-Object Text.StringBuilder; $last=Get-Date
  while(((Get-Date)-$last).TotalMilliseconds -lt $ms){
    $d=ReadSer; if($d){[void]$sb.Append($d); $last=Get-Date}; Start-Sleep -Milliseconds 100 }
  return $sb.ToString()
}

Write-Output "STAGE 0: trying to reboot the board ourselves via SysRq over serial break"
# If the kernel is alive (e.g. hung in rootwait) SysRq may still work even
# though it did not when systemd was running. Costs 15s to find out.
function SysRq([string]$k){
  try{
    $sp.BreakState = $true; Start-Sleep -Milliseconds 400
    $sp.BreakState = $false; Start-Sleep -Milliseconds 80
    $sp.Write($k)
  }catch{ Write-Output "   break failed: $($_.Exception.Message)" }
  Start-Sleep -Milliseconds 900
  return (Settle 1200)
}
$probe = SysRq 'h'
if($probe -match 'SysRq|sysrq|HELP'){
  Write-Output "   SysRq IS enabled - syncing and rebooting"
  [void](SysRq 's'); Start-Sleep -Seconds 2
  [void](SysRq 'b')
  Write-Output "   reboot requested"
} else {
  Write-Output "   SysRq not responding (no help text) - a manual power-cycle will be needed"
}

Write-Output "STAGE 1: spamming ESC -- POWER-CYCLE THE JETSON NOW"
$acc=''; $found=$false
$end=(Get-Date).AddSeconds(21600)
while((Get-Date) -lt $end){
  try{ $sp.Write($ESC) }catch{}
  Start-Sleep -Milliseconds 200
  $d=ReadSer; if($d){ $acc+=$d; if($acc.Length -gt 20000){$acc=$acc.Substring($acc.Length-8000)} }
  if($acc -match 'Select Language' -and $acc -match 'Boot Manager'){ $found=$true; break }
}
if(-not $found){ Write-Output "FAIL: never saw the UEFI menu"; $sp.Close(); exit 1 }
Write-Output "STAGE 1 OK: UEFI Setup"
[void](Settle 2000)

Write-Output "STAGE 2: -> Boot Manager"
$sp.Write($ESC+"[B"); Start-Sleep -Milliseconds 600
$sp.Write($ESC+"[B"); Start-Sleep -Milliseconds 600
$sp.Write("`r");      Start-Sleep -Milliseconds 1800
$s=Settle 2200
if($s -notmatch 'Boot Manager Menu'){ Write-Output "FAIL: no Boot Manager Menu"; $sp.Close(); exit 1 }
Write-Output "STAGE 2 OK"

Write-Output "STAGE 3: -> UEFI Shell"
for($i=1;$i -le 5;$i++){ $sp.Write($ESC+"[B"); Start-Sleep -Milliseconds 550; [void](ReadSer) }
$sp.Write("`r")
$s=Settle 6000
$sp.Write($ESC); Start-Sleep -Milliseconds 800     # skip startup.nsh countdown
[void](Settle 2500)
$sp.Write("`r`n"); Start-Sleep -Milliseconds 800
[void](Settle 1500)
Write-Output "STAGE 3 OK (shell reached)"

Write-Output "STAGE 4: boot with rdinit=/bin/sh (kernel-level; NVIDIA's /init cannot ignore it)"
# NVIDIA's initrd is a custom script that honours neither break= nor init=.
# rdinit= is handled by the kernel itself, so it runs our shell instead of /init.
$cmd='FS2:\boot\Image initrd=\boot\initrd root=PARTUUID=21c0c40f-8221-4367-8f06-281d50c5d741 rw rootwait rootfstype=ext4 console=ttyTCU0,115200 fbcon=map:0 video=efifb:off rdinit=/bin/sh'
Write-Output $cmd
$sp.Write($cmd+"`r`n")

$gotShell=$false
$deadline=(Get-Date).AddSeconds(180)
while((Get-Date) -lt $deadline){
  [void](ReadSer)
  Start-Sleep -Milliseconds 1000
  $sp.Write("`r`n"); Start-Sleep -Milliseconds 300
  $sp.Write('echo MARK_$((6*7))_END' + "`r`n")
  $r = Settle 2500
  if($r -match 'MARK_42_END'){ $gotShell=$true; break }
}
if(-not $gotShell){
  Write-Output "FAIL: rdinit gave no shell. Tail:"
  Write-Output (((Settle 3000) -split "`n" | Where-Object { $_ -match '\S' } | Select-Object -Last 30) -join "`n")
  $sp.Close(); exit 1
}
Write-Output "STAGE 4 OK: INITRAMFS SHELL (rdinit) CONFIRMED"

Write-Output "STAGE 5: load storage modules and mount the real root"
$M='/lib/modules/6.8.12-1021-tegra/kernel/drivers'
$prep = @(
  @("insmod $M/phy/tegra/phy-tegra194-p2u.ko; echo RC_phy=`$?",          'RC_phy='),
  @("insmod $M/pci/controller/dwc/pcie-tegra194.ko; echo RC_pcie=`$?",   'RC_pcie='),
  @('sleep 3; echo RC_settle=$?',                                        'RC_settle='),
  @("insmod $M/nvme/host/nvme.ko; echo RC_nvme=`$?",                     'RC_nvme='),
  @('sleep 3; ls /dev/nvme0n1p1; echo RC_dev=$?',                        'RC_dev='),
  @('mkdir -p /mnt; mount -t ext4 /dev/nvme0n1p1 /mnt; echo RC_mount=$?','RC_mount='),
  @('ls -d /mnt/etc/systemd/system; echo RC_check=$?',                   'RC_check=')
)
$mounted=$true
foreach($pair in $prep){
  $sp.Write($pair[0] + "`r`n")
  $out = Settle 3000
  if($out -match ([regex]::Escape($pair[1]) + '(\d+)')){
    $rc=$Matches[1]
    Write-Output ("  [{0}] {1}" -f $(if($rc -eq '0'){'ok'}else{"rc=$rc"}), $pair[0])
    if($rc -ne '0' -and $pair[1] -match 'RC_(mount|dev|check)='){ $mounted=$false }
  } else { Write-Output ("  [NO RESPONSE] " + $pair[0]); $mounted=$false }
}
if(-not $mounted){
  Write-Output "FAIL: could not mount the real root; not applying changes."
  $sp.Close(); exit 1
}
Write-Output "STAGE 5 OK: real root mounted at /mnt"

Write-Output "STAGE 6: applying the OOBE bypass to /mnt, then continuing the boot"
$fix = @(
  @('mkdir -p /mnt/etc/systemd/system/multi-user.target.wants; echo RC_mkdir=$?','RC_mkdir='),
  @('ln -sf /dev/null /mnt/etc/systemd/system/nv-oobe.service; echo RC_mask=$?','RC_mask='),
  @('ln -sf /usr/lib/systemd/system/ssh.service /mnt/etc/systemd/system/multi-user.target.wants/ssh.service; echo RC_ssh=$?','RC_ssh='),
  @('ln -sf /usr/lib/systemd/system/multi-user.target /mnt/etc/systemd/system/default.target; echo RC_target=$?','RC_target='),
  @('chroot /mnt /usr/sbin/useradd -m -s /bin/bash jetson; echo RC_useradd=$?','RC_useradd='),
  @('echo jetson:jetson | chroot /mnt /usr/sbin/chpasswd; echo RC_passwd=$?','RC_passwd='),
  @('chroot /mnt /usr/sbin/usermod -aG sudo,video,audio,dialout,plugdev jetson; echo RC_groups=$?','RC_groups='),
  @('chroot /mnt /usr/bin/id jetson; echo RC_id=$?','RC_id='),
  @('ls -l /mnt/etc/systemd/system/nv-oobe.service; echo RC_v=$?','RC_v='),
  @('sync; umount /mnt; echo RC_umount=$?','RC_umount=')
)
foreach($pair in $fix){
  $sp.Write($pair[0] + "`r`n")
  $out = Settle 3000
  if($out -match ([regex]::Escape($pair[1]) + '(\d+)')){
    $rc=$Matches[1]
    Write-Output ("  [{0}] {1}" -f $(if($rc -eq '0'){'ok'}else{"rc=$rc"}), $pair[0])
    $extra = (($out -split "`n" | Where-Object { $_ -match '\S' -and $_ -notmatch '^\[\s*\d' -and $_ -notmatch 'RC_' -and $_ -notmatch [regex]::Escape($pair[0]) }) -join ' | ')
    if($extra){ Write-Output ("        " + $extra) }
  } else { Write-Output ("  [NO RESPONSE] " + $pair[0]) }
}

Write-Output "STAGE 6b: handing control back to NVIDIA's /init to finish booting"
$sp.Write("exec /init`r`n")

Write-Output "STAGE 7: waiting for a login prompt (up to 5 min)"
$gotLogin=$false
$deadline=(Get-Date).AddSeconds(300)
$acc=''
while((Get-Date) -lt $deadline){
  $d=ReadSer; if($d){ $acc+=$d; if($acc.Length -gt 30000){$acc=$acc.Substring($acc.Length-10000)} }
  if($acc -match '[Ll]ogin:\s*$' -or $acc -match '\slogin:\s'){ $gotLogin=$true; break }
  Start-Sleep -Milliseconds 300
}
if(-not $gotLogin){
  Write-Output "  no login prompt seen; nudging"
  $sp.Write("`r`n"); $acc += (Settle 4000)
  if($acc -match 'login:'){ $gotLogin=$true }
}

if(-not $gotLogin){
  Write-Output "FAIL: never reached a login prompt. Tail:"
  Write-Output ((($acc -split "`n" | Where-Object { $_ -match '\S' } | Select-Object -Last 30) -join "`n"))
  $sp.Close(); exit 1
}
Write-Output "STAGE 7 OK: login prompt reached (nv-oobe is bypassed)"

Write-Output "STAGE 8: logging in as jetson"
$sp.Write("jetson`r`n"); Start-Sleep -Milliseconds 1500
[void](Settle 2000)
$sp.Write("jetson`r`n"); Start-Sleep -Milliseconds 2500
$r = Settle 3000
# confirm a real shell again with arithmetic the tty cannot fake
$sp.Write('echo LOGIN_$((8*9))_OK' + "`r`n")
$r2 = Settle 3000
if($r2 -notmatch 'LOGIN_72_OK'){
  Write-Output "FAIL: login did not yield a shell. Tail:"
  Write-Output ((($r + $r2) -split "`n" | Where-Object { $_ -match '\S' } | Select-Object -Last 25) -join "`n")
  $sp.Close(); exit 1
}
Write-Output "STAGE 8 OK: logged in, real shell confirmed"

Write-Output "STAGE 9: bringing up networking and sshd"
$net = @(
  'sudo -S true <<< jetson 2>/dev/null; echo RC_sudo=$?',
  'echo jetson | sudo -S systemctl enable --now ssh 2>&1 | tail -2; echo RC_ssh=$?',
  'echo jetson | sudo -S systemctl is-active ssh; echo RC_active=$?',
  'for i in $(ls /sys/class/net | grep -v lo); do echo jetson | sudo -S ip link set $i up; done; echo RC_up=$?',
  'echo jetson | sudo -S nmcli networking on 2>/dev/null; echo RC_nm=$?',
  'sleep 12; echo RC_wait=$?',
  'ip -4 -br addr; echo RC_addr=$?',
  'ip route | head -3; echo RC_route=$?',
  'ss -ltn | head -8; echo RC_ports=$?',
  'hostname -I; echo RC_ip=$?'
)
foreach($c in $net){
  $sp.Write($c + "`r`n")
  $out = Settle 3500
  Write-Output ("  $ " + $c)
  $clean = ($out -split "`n" | Where-Object { $_ -match '\S' -and $_ -notmatch [regex]::Escape($c) }) -join "`n"
  if($clean){ Write-Output $clean }
}

Write-Output "### STAGE 9 complete - look for an IPv4 address above; SSH with: ssh jetson@<addr> ###"
$sp.Close()
