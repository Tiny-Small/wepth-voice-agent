param(
    [Parameter(Position=0)][string]$Operation,
    [Parameter(Position=1)][string]$Query = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
try { Add-Type -AssemblyName Microsoft.VisualBasic } catch { }
try { Add-Type -AssemblyName System.Runtime.WindowsRuntime } catch { }

function Get-SpotifyWindow {
    $root = [System.Windows.Automation.AutomationElement]::RootElement
    $windows = $root.FindAll([System.Windows.Automation.TreeScope]::Children,
        [System.Windows.Automation.Condition]::TrueCondition)
    foreach ($window in $windows) {
        try {
            $owner = Get-Process -Id $window.Current.ProcessId -ErrorAction SilentlyContinue
            if ($null -ne $owner -and $owner.ProcessName -eq "Spotify" -and
                $window.Current.NativeWindowHandle -ne 0) { return $window }
        } catch { }
    }
    throw "Spotify window not found"
}

function Focus-SpotifyWindow($root) {
    # Spotify exposes a Chromium omnibox as its only UIA Edit element. It is
    # read-only and cannot be used to enter the application's search query.
    # Foreground the real window, then use Spotify's Search button and keyboard.
    try {
        $process = Get-Process -Name Spotify -ErrorAction SilentlyContinue |
            Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
        if ($null -ne $process) {
            [Microsoft.VisualBasic.Interaction]::AppActivate($process.Id) | Out-Null
        }
    } catch { }
    try { $root.SetFocus() } catch { }
}

function Get-SpotifySearchButton($root) {
    $condition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, "Search")
    return $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
}

function Get-SpotifySearchResultsGrid($root) {
    $condition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, "Search results")
    $grid = $root.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $condition)
    if ($null -eq $grid -or $grid.Current.ControlType.ProgrammaticName -ne "ControlType.DataGrid") {
        return $null
    }
    return $grid
}

function Get-SpotifyPlayButtons($grid) {
    $condition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Button)
    return $grid.FindAll([System.Windows.Automation.TreeScope]::Descendants, $condition)
}

function Send-SpotifySearchQuery($text) {
    try {
        [System.Windows.Forms.Clipboard]::SetText($text)
        [System.Windows.Forms.SendKeys]::SendWait("^a")
        [System.Windows.Forms.SendKeys]::SendWait("^v")
    } catch {
        # Clipboard access can be unavailable in a non-interactive desktop.
        # Escape SendKeys metacharacters as a bounded fallback.
        $escaped = $text -replace '([+^%~(){}])', '{$1}'
        [System.Windows.Forms.SendKeys]::SendWait($escaped)
    }
}

function Wait-SpotifySearchResults {
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        $root = Get-SpotifyWindow
        $grid = Get-SpotifySearchResultsGrid $root
        if ($null -ne $grid) {
            $nodes = $grid.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                [System.Windows.Automation.Condition]::TrueCondition)
            if ($nodes.Count -gt 0) { return $grid }
        }
        Start-Sleep -Milliseconds 200
    }
    throw "Spotify search results did not become ready"
}

function Invoke-SpotifyTopResult($query) {
    $root = Get-SpotifyWindow
    $container = Get-SpotifySearchResultsGrid $root
    if ($null -eq $container) { throw "Spotify search-results container not found" }
    $items = Get-SpotifyPlayButtons $container
    $item = $null
    $fallback = $null
    foreach ($candidate in $items) {
        try {
            $name = $candidate.Current.Name
            if ($candidate.Current.IsEnabled -and $name -and $name -match "(?i)^Play(?: |$)") {
                if ($null -eq $fallback) { $fallback = $candidate }
                if ($query -and $name -match [regex]::Escape($query)) { $item = $candidate; break }
            }
        } catch { }
    }
    if ($null -ne $item -or $null -ne $fallback) {
        if ($null -eq $item) { $item = $fallback }
        $item.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
        return
    }

    # A broad term such as "Jazz" can first resolve to a genre card rather
    # than a directly playable row. Open that card, then choose a playable
    # result from the genre page.
    $linkCondition = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Hyperlink)
    $links = $container.FindAll([System.Windows.Automation.TreeScope]::Descendants, $linkCondition)
    $link = $null
    foreach ($candidate in $links) {
        if ($candidate.Current.IsEnabled -and $candidate.Current.Name -ieq $query) {
            $link = $candidate
            break
        }
    }
    if ($null -eq $link) { throw "Spotify playable search result not found" }
    $link.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    Start-Sleep -Milliseconds 500
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        $playCondition = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
            [System.Windows.Automation.ControlType]::Button)
        $buttons = $root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $playCondition)
        $fallback = $null
        foreach ($candidate in $buttons) {
            $name = $candidate.Current.Name
            if ($candidate.Current.IsEnabled -and $name -match "(?i)^Play ") {
                if ($null -eq $fallback) { $fallback = $candidate }
                if ($name -match [regex]::Escape($query)) { $candidate.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke(); return }
            }
        }
        if ($null -ne $fallback) { $fallback.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke(); return }
        Start-Sleep -Milliseconds 200
    }
    throw "Spotify playable result did not become ready"
}

function Test-SpotifySearchReady {
    $root = Get-SpotifyWindow
    $grid = Get-SpotifySearchResultsGrid $root
    if ($null -eq $grid) { return $false }
    $nodes = $grid.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition)
    return ($nodes.Count -gt 0)
}

function Await-WinRT($operation, [Type]$resultType) {
    # WinRT proxy objects in Windows PowerShell 5 do not reliably expose their
    # IAsyncOperation<T> interface through GetInterfaces(). Select the documented
    # AsTask(IAsyncOperation<T>) overload and supply T explicitly instead.
    $type = [System.WindowsRuntimeSystemExtensions]
    # Windows PowerShell reports the WinRT projection parameter by its short
    # name; its reflected FullName varies between Windows builds.
    $method = $type.GetMethods() | Where-Object {
        $_.Name -eq "AsTask" -and $_.IsGenericMethodDefinition `
            -and $_.GetGenericArguments().Count -eq 1 `
            -and $_.GetParameters().Count -eq 1 `
            -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    } | Select-Object -First 1
    if ($null -eq $method) { throw "WinRT AsTask bridge is unavailable" }
    if ($null -eq $resultType) { throw "WinRT operation result type is required" }
    $task = $method.MakeGenericMethod($resultType).Invoke($null, @($operation))
    return $task.GetAwaiter().GetResult()
}

function Get-SpotifySession {
    $manager = Await-WinRT ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager, Windows.Media.Control, ContentType = WindowsRuntime]::RequestAsync()) ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager, Windows.Media.Control, ContentType = WindowsRuntime])
    $session = $manager.GetSessions() | Where-Object { $_.SourceAppUserModelId -match "Spotify" } | Select-Object -First 1
    if ($null -eq $session) { throw "Spotify media session not found" }
    return $session
}

switch ($Operation) {
    "open" {
        Start-Process "spotify:"
        Start-Sleep -Milliseconds 700
        @{ running = [bool](Get-Process -Name Spotify -ErrorAction SilentlyContinue) } | ConvertTo-Json -Compress
    }
    "search" {
        Start-Process "spotify:"
        Start-Sleep -Milliseconds 500
        $root = Get-SpotifyWindow
        Focus-SpotifyWindow $root
        $button = Get-SpotifySearchButton $root
        if ($null -eq $button) { throw "Spotify Search button not found" }
        $button.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
        Start-Sleep -Milliseconds 350
        Focus-SpotifyWindow $root
        Send-SpotifySearchQuery $Query
        [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
        Wait-SpotifySearchResults | Out-Null
        @{ running = $true; search_query = $Query; search_results = $Query; search_ready = $true } | ConvertTo-Json -Compress
    }
    "play" {
        Invoke-SpotifyTopResult $Query
        @{ running = $true } | ConvertTo-Json -Compress
    }
    "pause" { $ok = Await-WinRT ((Get-SpotifySession).TryPauseAsync()) ([bool]); if (-not $ok) { throw "Spotify pause was rejected" }; @{ running = $true } | ConvertTo-Json -Compress }
    "resume" { $ok = Await-WinRT ((Get-SpotifySession).TryPlayAsync()) ([bool]); if (-not $ok) { throw "Spotify resume was rejected" }; @{ running = $true } | ConvertTo-Json -Compress }
    "skip" { $ok = Await-WinRT ((Get-SpotifySession).TrySkipNextAsync()) ([bool]); if (-not $ok) { throw "Spotify skip was rejected" }; @{ running = $true } | ConvertTo-Json -Compress }
    "observe" {
        $running = [bool](Get-Process -Name Spotify -ErrorAction SilentlyContinue)
        $status = "unknown"
        $source = "Spotify"
        $title = $null
        $artist = $null
        try {
            $session = Get-SpotifySession
            $source = $session.SourceAppUserModelId
            $status = $session.GetPlaybackInfo().PlaybackStatus.ToString()
            $properties = Await-WinRT ($session.TryGetMediaPropertiesAsync()) ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionMediaProperties, Windows.Media.Control, ContentType = WindowsRuntime])
            $title = $properties.Title
            $artist = $properties.Artist
        } catch {
            if ($_.Exception.Message -notmatch "media session not found") { throw }
            $status = "unknown"
        }
        try { $searchReady = Test-SpotifySearchReady } catch {
            if ($running) { throw }
            $searchReady = $false
        }
        @{ running = $running; focused = $false; status = $status; source = $source; title = $title; artist = $artist;
           search_query = $null; search_results = $null; search_ready = $searchReady } | ConvertTo-Json -Compress
    }
    default { throw "unknown Spotify operation: $Operation" }
}
