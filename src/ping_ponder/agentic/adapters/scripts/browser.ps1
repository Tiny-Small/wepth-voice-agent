param(
    [Parameter(Position=0)][string]$Operation,
    [Parameter(Position=1)][string]$Target = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
try { Add-Type -AssemblyName Microsoft.VisualBasic } catch { }

function Get-ChromeWindow {
    $root = [System.Windows.Automation.AutomationElement]::RootElement
    $condition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ClassNameProperty, "Chrome_WidgetWin_1")
    $windows = $root.FindAll([System.Windows.Automation.TreeScope]::Children, $condition)
    foreach ($window in $windows) {
        $owner = Get-Process -Id $window.Current.ProcessId -ErrorAction SilentlyContinue
        if ($null -ne $owner -and $owner.ProcessName -eq "chrome") { return $window }
    }
    throw "Chrome window not found"
}

function Ensure-ChromeWindow {
    # Commands normally reuse the already-running active Chrome window. Launch
    # only when no Chrome window exists (for direct adapter calls without the
    # planner's OpenBrowser prerequisite).
    try { return (Get-ChromeWindow) } catch { }
    Start-Process "chrome.exe"
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try { return (Get-ChromeWindow) } catch { Start-Sleep -Milliseconds 100 }
    }
    throw "Chrome window not found after launch"
}

function Get-ChromeAddressElement($root) {
    # Chrome has used both the omnibox automation id and localized UIA names
    # across versions. Prefer the stable id, then the names exposed by the
    # address-bar control. The keyboard path in Set-ChromeAddress remains the
    # final fallback when UIA does not expose the edit control at all.
    $condition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::AutomationIdProperty, "omnibox")
    $address = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
    if ($null -ne $address) { return $address }
    foreach ($name in @("Address and search bar", "Address bar")) {
        $condition = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::NameProperty, $name)
        $address = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
        if ($null -ne $address) { return $address }
    }
    return $null
}

function Get-ChromeAddress {
    $root = Get-ChromeWindow
    $address = Get-ChromeAddressElement $root
    if ($null -eq $address) { return $null }
    try {
        return $address.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern).Current.Value
    } catch { return $null }
}

function Focus-ChromeWindow($root) {
    try {
        $process = Get-Process -Name chrome -ErrorAction SilentlyContinue |
            Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
        if ($null -ne $process) {
            [Microsoft.VisualBasic.Interaction]::AppActivate($process.Id) | Out-Null
        }
    } catch { }
    try { $root.SetFocus() } catch { }
}

function Set-ChromeAddress($text) {
    $root = Get-ChromeWindow
    $address = Get-ChromeAddressElement $root
    if ($null -ne $address) {
        try {
            $address.SetFocus()
            $address.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern).SetValue($text)
            return
        } catch {
            # UIA can expose the control but reject focus/value writes while
            # Chrome is changing tabs. Fall through to the foreground keyboard path.
        }
    }

    Focus-ChromeWindow $root
    [System.Windows.Forms.SendKeys]::SendWait("^l")
    Start-Sleep -Milliseconds 100
    # Escape SendKeys metacharacters so the complete URL remains literal.
    $escaped = $text -replace '([+^%~(){}])', '{$1}'
    [System.Windows.Forms.SendKeys]::SendWait($escaped)
}

function Get-ChromeButtonEnabled($name) {
    $root = Get-ChromeWindow
    $condition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, $name)
    $button = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
    return ($null -ne $button -and $button.Current.IsEnabled)
}

function Invoke-ChromeButton($name) {
    $root = Get-ChromeWindow
    $condition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, $name)
    $button = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
    if ($null -eq $button) { throw "Chrome $name button not found" }
    $button.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
}

switch ($Operation) {
    "open" {
        Start-Process "chrome.exe"
        Start-Sleep -Milliseconds 700
        @{ running = [bool](Get-Process -Name chrome -ErrorAction SilentlyContinue) } | ConvertTo-Json -Compress
    }
    "search" {
        $root = Ensure-ChromeWindow
        Focus-ChromeWindow $root
        Set-ChromeAddress "https://www.google.com/search?q=$([uri]::EscapeDataString($Target))"
        [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
        @{ running = $true; search_query = $Target; search_results = $Target; search_ready = $true } | ConvertTo-Json -Compress
    }
    "navigate" {
        Set-ChromeAddress $Target
        [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
        @{ running = $true; navigation_target = $Target; current_url = $Target } | ConvertTo-Json -Compress
    }
    "back" { Invoke-ChromeButton "Back"; @{ running = $true } | ConvertTo-Json -Compress }
    "forward" { Invoke-ChromeButton "Forward"; @{ running = $true } | ConvertTo-Json -Compress }
    "observe" {
        $running = [bool](Get-Process -Name chrome -ErrorAction SilentlyContinue)
        try { $url = Get-ChromeAddress } catch {
            if ($running) { throw }
            $url = $null
        }
        try { $canBack = Get-ChromeButtonEnabled "Back"; $canForward = Get-ChromeButtonEnabled "Forward" } catch {
            if ($running) { throw }
            $canBack = $false; $canForward = $false
        }
        @{ running = $running; current_url = $url; navigation_target = $url; history = @(); history_index = -1;
           can_back = $canBack; can_forward = $canForward } | ConvertTo-Json -Compress
    }
    default { throw "unknown Browser operation: $Operation" }
}
