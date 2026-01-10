# Log-Überwachung für MCP-Server und Model Service
Write-Host "=== Log-Überwachung gestartet ===" -ForegroundColor Green
Write-Host "Drücken Sie Ctrl+C zum Beenden" -ForegroundColor Yellow
Write-Host ""

$mcpLog = "logs\mcp_server.log"
$modelLog = "logs\model_service.log"

# Initiale Log-Positionen
$mcpLastPos = 0
$modelLastPos = 0

if (Test-Path $mcpLog) {
    $mcpLastPos = (Get-Item $mcpLog).Length
}
if (Test-Path $modelLog) {
    $modelLastPos = (Get-Item $modelLog).Length
}

while ($true) {
    # Prüfe MCP-Server Logs
    if (Test-Path $mcpLog) {
        $currentSize = (Get-Item $mcpLog).Length
        if ($currentSize -gt $mcpLastPos) {
            $newContent = Get-Content $mcpLog -Tail 5
            Write-Host "[MCP-Server] $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
            $newContent | ForEach-Object { Write-Host "  $_" }
            Write-Host ""
            $mcpLastPos = $currentSize
        }
    }
    
    # Prüfe Model Service Logs
    if (Test-Path $modelLog) {
        $currentSize = (Get-Item $modelLog).Length
        if ($currentSize -gt $modelLastPos) {
            $newContent = Get-Content $modelLog | Select-String -Pattern "\[CHAT\]" | Select-Object -Last 3
            if ($newContent) {
                Write-Host "[Model Service] $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Magenta
                $newContent | ForEach-Object { Write-Host "  $_" }
                Write-Host ""
            }
            $modelLastPos = $currentSize
        }
    }
    
    Start-Sleep -Seconds 1
}
