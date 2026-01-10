# MCP Status-Check Script
Write-Host "=== MCP Status-Check ===" -ForegroundColor Green
Write-Host ""

# 1. Model Service Status
Write-Host "1. Model Service Status:" -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8001/status" -UseBasicParsing
    $status = $response.Content | ConvertFrom-Json
    Write-Host "   ✅ Model Service läuft" -ForegroundColor Green
    Write-Host "   ✅ Text Model: $($status.text_model.loaded) - $($status.text_model.model_id)" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Model Service nicht erreichbar" -ForegroundColor Red
}

Write-Host ""

# 2. MCP-Server Logs
Write-Host "2. MCP-Server Logs:" -ForegroundColor Cyan
$mcpLog = "logs\mcp_server.log"
if (Test-Path $mcpLog) {
    $lines = Get-Content $mcpLog
    Write-Host "   ✅ Log-Datei existiert ($($lines.Count) Zeilen)" -ForegroundColor Green
    Write-Host "   Letzte Einträge:" -ForegroundColor Yellow
    Get-Content $mcpLog -Tail 5 | ForEach-Object { Write-Host "     $_" }
    
    # Prüfe auf Requests
    $hasRequests = $lines | Select-String -Pattern "Request erhalten|Processing request|Tool-Aufruf"
    if ($hasRequests) {
        Write-Host "   ✅ Requests gefunden in Logs" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Keine Requests in Logs" -ForegroundColor Red
        Write-Host "   → Cursor ruft das Tool nicht auf" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ❌ Log-Datei existiert nicht" -ForegroundColor Red
    Write-Host "   → MCP-Server wurde noch nicht gestartet" -ForegroundColor Yellow
}

Write-Host ""

# 3. MCP-Konfiguration
Write-Host "3. MCP-Konfiguration:" -ForegroundColor Cyan
$mcpConfigPath = "$env:APPDATA\Cursor\User\globalStorage\mcp.json"
if (Test-Path $mcpConfigPath) {
    Write-Host "   ✅ MCP-Konfiguration gefunden" -ForegroundColor Green
    $config = Get-Content $mcpConfigPath | ConvertFrom-Json
    if ($config.mcpServers.'local-ai') {
        Write-Host "   ✅ 'local-ai' Server konfiguriert" -ForegroundColor Green
        Write-Host "   Command: $($config.mcpServers.'local-ai'.command)" -ForegroundColor Yellow
        Write-Host "   Args: $($config.mcpServers.'local-ai'.args -join ' ')" -ForegroundColor Yellow
    } else {
        Write-Host "   ❌ 'local-ai' Server nicht in Konfiguration" -ForegroundColor Red
    }
} else {
    Write-Host "   ❌ MCP-Konfiguration nicht gefunden" -ForegroundColor Red
    Write-Host "   Erwarteter Pfad: $mcpConfigPath" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Nächste Schritte ===" -ForegroundColor Green
Write-Host "1. Prüfen Sie Cursor MCP-Logs: View → Output → MCP" -ForegroundColor Yellow
Write-Host "2. Prüfen Sie Tools: Ctrl+Shift+P → 'MCP: Show Servers'" -ForegroundColor Yellow
Write-Host "3. Falls Server nicht sichtbar: Cursor neu starten" -ForegroundColor Yellow
