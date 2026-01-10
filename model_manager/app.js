// Model Manager Frontend - Kommuniziert direkt mit Model-Service (Port 8001)

const API_BASE = 'http://127.0.0.1:8001';
// REFRESH_INTERVAL entfernt - kein automatisches Polling mehr

// DOM Elements
const mainContent = document.getElementById('mainContent');
const serviceStatus = document.getElementById('serviceStatus');
const serviceStatusText = document.getElementById('serviceStatusText');

// DOM Elements
const btnRestart = document.getElementById('btnRestart');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkServiceStatus();
    loadModelStatus();
    loadMCPSettings();
    
    // Restart Button Event Listener
    if (btnRestart) {
        btnRestart.addEventListener('click', restartServer);
    }
    
    // MCP Settings Toggle Event Listener
    const autoModelSilentModeToggle = document.getElementById('autoModelSilentMode');
    if (autoModelSilentModeToggle) {
        autoModelSilentModeToggle.addEventListener('change', async (e) => {
            await saveMCPSettings(e.target.checked);
        });
    }
    
    // Max Length Settings werden in loadMCPSettings() geladen (nachdem Settings-Sektion angezeigt wurde)
    const maxLengthSlider = document.getElementById('maxLengthSlider');
    const maxLengthValue = document.getElementById('maxLengthValue');
    const btnSaveMaxLength = document.getElementById('btnSaveMaxLength');
    
    if (maxLengthSlider && maxLengthValue) {
        maxLengthSlider.addEventListener('input', (e) => {
            maxLengthValue.textContent = parseInt(e.target.value);
        });
    }
    
    if (btnSaveMaxLength) {
        btnSaveMaxLength.addEventListener('click', async () => {
            await saveMaxLengthSettings();
        });
    }
    
    // Kein automatisches Polling mehr - Status wird nur bei Bedarf aktualisiert
    // (z.B. nach Modell-Operationen, manueller Refresh, etc.)
});

// Prüfe Service-Status
async function checkServiceStatus() {
    try {
        const response = await fetch(`${API_BASE}/`);
        if (response.ok) {
            serviceStatus.className = 'status-indicator connected';
            serviceStatusText.textContent = 'Verbunden';
        } else {
            throw new Error('Service nicht erreichbar');
        }
    } catch (error) {
        serviceStatus.className = 'status-indicator disconnected';
        serviceStatusText.textContent = 'Nicht verbunden';
    }
}

// Lade Modell-Status
async function loadModelStatus(preserveSelects = false) {
    try {
        const response = await fetch(`${API_BASE}/status`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const status = await response.json();
        renderModelManager(status, preserveSelects);
        
        // Update Service-Status
        serviceStatus.className = 'status-indicator connected';
        serviceStatusText.textContent = 'Verbunden';
    } catch (error) {
        console.error('Fehler beim Laden des Model-Status:', error);
        mainContent.innerHTML = `
            <div class="error">
                <h3>Fehler beim Laden des Model-Status</h3>
                <p>${error.message}</p>
                <p style="margin-top: 10px; font-size: 14px; color: var(--text-secondary);">
                    Stellen Sie sicher, dass der Model-Service läuft (Port 8001).
                </p>
            </div>
        `;
        
        serviceStatus.className = 'status-indicator disconnected';
        serviceStatusText.textContent = 'Nicht verbunden';
    }
}

// Lade MCP Settings
async function loadMCPSettings() {
    try {
        const response = await fetch(`${API_BASE}/mcp/settings`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const settings = await response.json();
        const toggle = document.getElementById('autoModelSilentMode');
        const settingsSection = document.getElementById('settingsSection');
        
        if (toggle) {
            toggle.checked = settings.auto_model_silent_mode || false;
        }
        
        if (settingsSection) {
            settingsSection.style.display = 'block';
            // Lade max_length Settings nachdem Settings-Sektion angezeigt wurde
            await loadMaxLengthSettings();
        }
    } catch (error) {
        console.error('Fehler beim Laden der MCP-Einstellungen:', error);
        // Zeige Settings-Sektion trotzdem an (mit Default-Werten)
        const settingsSection = document.getElementById('settingsSection');
        if (settingsSection) {
            settingsSection.style.display = 'block';
            // Versuche trotzdem max_length Settings zu laden
            await loadMaxLengthSettings();
        }
    }
}

// Speichere MCP Settings
// Max Length Settings
async function loadMaxLengthSettings() {
    try {
        const response = await fetch(`${API_BASE}/settings/max-length`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        const maxLengthSlider = document.getElementById('maxLengthSlider');
        const maxLengthValue = document.getElementById('maxLengthValue');
        
        if (maxLengthSlider && maxLengthValue) {
            maxLengthSlider.min = data.min;
            maxLengthSlider.max = data.max;
            maxLengthSlider.value = data.max_length;
            maxLengthValue.textContent = data.max_length;
        }
    } catch (error) {
        console.error('Fehler beim Laden der max_length Settings:', error);
    }
}

async function saveMaxLengthSettings() {
    const maxLengthSlider = document.getElementById('maxLengthSlider');
    if (!maxLengthSlider) {
        return;
    }
    
    const maxLength = parseInt(maxLengthSlider.value);
    
    try {
        const response = await fetch(`${API_BASE}/settings/max-length`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ max_length: maxLength })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        const data = await response.json();
        alert(`Max. Länge erfolgreich auf ${data.max_length} gesetzt!`);
        
        // Aktualisiere UI
        const maxLengthValue = document.getElementById('maxLengthValue');
        if (maxLengthValue) {
            maxLengthValue.textContent = data.max_length;
        }
    } catch (error) {
        console.error('Fehler beim Speichern der max_length Settings:', error);
        alert(`Fehler beim Speichern: ${error.message}`);
    }
}

async function saveMCPSettings(autoModelSilentMode) {
    try {
        const response = await fetch(`${API_BASE}/mcp/settings`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                auto_model_silent_mode: autoModelSilentMode
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        console.log('MCP-Einstellungen gespeichert:', result);
        
        // Zeige Bestätigung
        const toggle = document.getElementById('autoModelSilentMode');
        if (toggle) {
            const status = autoModelSilentMode ? 'aktiviert' : 'deaktiviert';
            // Kurze visuelle Bestätigung (optional)
            toggle.style.transition = 'all 0.3s';
            setTimeout(() => {
                toggle.style.transition = '';
            }, 300);
        }
    } catch (error) {
        console.error('Fehler beim Speichern der MCP-Einstellungen:', error);
        alert(`Fehler beim Speichern: ${error.message}`);
        
        // Setze Toggle zurück auf vorherigen Wert
        await loadMCPSettings();
    }
}

// Rendere Model Manager UI
function renderModelManager(status, preserveSelects = false) {
    // Speichere aktuelle Select-Werte wenn preserveSelects true ist
    const selectValues = {};
    if (preserveSelects) {
        document.querySelectorAll('.model-select').forEach(select => {
            selectValues[select.id] = select.value;
        });
    }
    
    const html = `
        <div class="model-section">
            <h2>Text-Modelle</h2>
            ${renderModelCard('text', status.text_model)}
        </div>
        
        <div class="model-section">
            <h2>Audio-Modelle</h2>
            ${renderModelCard('audio', status.audio_model)}
        </div>
        
        <div class="model-section">
            <h2>Image-Modelle</h2>
            ${renderModelCard('image', status.image_model)}
        </div>
    `;
    mainContent.innerHTML = html;
    
    // Stelle Select-Werte wieder her
    if (preserveSelects) {
        Object.keys(selectValues).forEach(selectId => {
            const select = document.getElementById(selectId);
            if (select && selectValues[selectId]) {
                select.value = selectValues[selectId];
            }
        });
    }
    
    // Event Listeners für Buttons
    document.querySelectorAll('.btn-on').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const modelType = e.target.dataset.modelType;
            // Nutze immer das Select, nicht das data-model-id (kann "Kein Modell geladen" sein)
            const select = document.getElementById(`modelSelect_${modelType}`);
            const modelId = select && select.value ? select.value : null;
            if (!modelId) {
                alert('Bitte wählen Sie zuerst ein Modell aus dem Dropdown aus');
                return;
            }
            await loadModel(modelType, modelId);
        });
    });
    
    document.querySelectorAll('.btn-off').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const modelType = e.target.dataset.modelType;
            await unloadModel(modelType);
        });
    });
}

// Rendere Model Card
function renderModelCard(modelType, modelStatus) {
    const loaded = modelStatus.loaded;
    const modelId = modelStatus.model_id || 'Kein Modell geladen';
    const activeClients = modelStatus.active_clients || [];
    
    const statusIcon = loaded ? '●' : '○';
    const statusClass = loaded ? 'loaded' : 'not-loaded';
    const statusText = loaded ? 'Geladen' : 'Nicht geladen';
    
    // Lade verfügbare Modelle für diesen Typ
    const availableModels = getAvailableModels(modelType);
    
    const modelSelectHtml = availableModels.length > 0 ? `
        <select id="modelSelect_${modelType}" class="model-select" style="width: 100%; padding: 8px; margin-top: 10px; background: var(--bg-secondary); color: var(--text-primary); border: 1px solid var(--border); border-radius: 4px;">
            <option value="">-- Modell auswählen --</option>
            ${availableModels.map(m => `<option value="${m.id}" ${m.id === modelId ? 'selected' : ''}>${m.name}</option>`).join('')}
        </select>
    ` : '<p style="color: var(--text-secondary); font-size: 13px; margin-top: 10px;">Keine Modelle verfügbar</p>';
    
    const clientsHtml = activeClients.length > 0
        ? activeClients.map(client => `
            <div class="client-item">
                <span class="client-name">${client.app_name}</span>
                <span class="client-time">seit ${formatTime(client.active_since)}</span>
            </div>
        `).join('')
        : '<div class="no-clients">(keine aktiven Apps)</div>';
    
    return `
        <div class="model-card">
            <div class="model-header">
                <div class="model-info">
                    <h3>${modelId}</h3>
                    <div class="model-status">
                        <span class="status-icon ${statusClass}">${statusIcon}</span>
                        <span class="status-text">${statusText}</span>
                    </div>
                </div>
                <div class="model-actions">
                    <button class="btn btn-on" data-model-type="${modelType}" 
                            ${loaded ? 'disabled' : ''} id="btnOn_${modelType}">
                        ON
                    </button>
                    <button class="btn btn-off" data-model-type="${modelType}"
                            ${!loaded ? 'disabled' : ''} id="btnOff_${modelType}">
                        OFF
                    </button>
                </div>
            </div>
            
            ${modelSelectHtml}
            
            <div class="clients-section">
                <h4>Aktive Apps:</h4>
                <div class="client-list">
                    ${clientsHtml}
                </div>
            </div>
        </div>
    `;
}

// Lade verfügbare Modelle (aus Config - für jetzt hardcoded, später über API)
function getAvailableModels(modelType) {
    // TODO: Lade verfügbare Modelle über API
    // Für jetzt: Hardcoded basierend auf config.json
    const models = {
        text: [
            { id: 'qwen-2.5-3b', name: 'Qwen 2.5 3B' },
            { id: 'qwen-2.5-7b-instruct', name: 'Qwen 2.5 7B Instruct ⭐' },
            { id: 'phi-3-mini-4k', name: 'Phi-3 Mini 4K Instruct' },
            { id: 'mistral-7b-instruct', name: 'Mistral 7B Instruct' }
        ],
        audio: [
            { id: 'whisper-large-v3', name: 'Whisper Large V3' }
        ],
        image: [
            { id: 'sdxl-base-1.0', name: 'Stable Diffusion XL Base 1.0 ⭐' }
        ]
    };
    return models[modelType] || [];
}

// Formatiere Zeit
// PollModelLoadStatus entfernt - kein automatisches Polling mehr während des Modellladens

function formatTime(seconds) {
    if (seconds < 60) {
        return `${seconds} Sek`;
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        return `${minutes} Min`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }
}

// Lade Modell
async function loadModel(modelType, modelId) {
    // Wenn kein modelId, nutze Select
    if (!modelId) {
        const select = document.getElementById(`modelSelect_${modelType}`);
        if (select && select.value) {
            modelId = select.value;
        } else {
            alert('Bitte wählen Sie ein Modell aus');
            return;
        }
    }
    
    try {
        const response = await fetch(`${API_BASE}/models/${modelType}/load`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_id: modelId })
        });
        
        if (!response.ok) {
            // Versuche JSON zu parsen, falls es HTML ist, nutze Text
            let error;
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                error = await response.json();
            } else {
                const text = await response.text();
                error = { detail: `HTTP ${response.status}: ${text.substring(0, 100)}` };
            }
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        // Wenn Modell geladen wird, zeige Meldung (kein Polling mehr)
        if (result.status === "loading") {
            alert(`Modell wird geladen: ${result.model_id}\nBitte aktualisieren Sie die Seite später, um den Status zu prüfen.`);
        } else if (result.status === "success") {
            alert(`Modell ${result.model_id} ist bereits geladen!`);
        }
        
        // Lade Status neu (einmalig)
        await loadModelStatus();
    } catch (error) {
        console.error(`Fehler beim Laden des ${modelType}-Modells:`, error);
        alert(`Fehler beim Laden des Modells: ${error.message}`);
    }
}

// Entlade Modell
async function unloadModel(modelType) {
    try {
        const response = await fetch(`${API_BASE}/models/${modelType}/unload`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        // Lade Status neu
        await loadModelStatus();
    } catch (error) {
        console.error(`Fehler beim Entladen des ${modelType}-Modells:`, error);
        alert(`Fehler beim Entladen des Modells: ${error.message}`);
    }
}

// Server Restart
async function restartServer() {
    if (!confirm('Möchten Sie den Server wirklich neu starten? Dies wird einige Sekunden dauern.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/restart`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        
        // Zeige Status-Update
        if (serviceStatusText) {
            serviceStatusText.textContent = 'Server startet neu...';
            serviceStatusText.style.color = 'var(--warning)';
        }
        
        // Warte auf Server-Neustart und aktualisiere Status
        await waitForServerRestart();
        
        // Status aktualisieren
        await checkServiceStatus();
        await loadModelStatus();
        
        alert('Server wurde erfolgreich neu gestartet.');
    } catch (error) {
        console.error('Fehler beim Server-Restart:', error);
        alert('Fehler beim Server-Restart: ' + error.message);
        // Status zurücksetzen
        await checkServiceStatus();
    }
}

// Wartet auf Server-Neustart
async function waitForServerRestart() {
    const maxAttempts = 30; // 30 Versuche = ca. 30 Sekunden
    const delay = 1000; // 1 Sekunde zwischen Versuchen
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            // Versuche Status-Endpoint zu erreichen mit Timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 2000);
            
            const response = await fetch(`${API_BASE}/status`, {
                method: 'GET',
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (response.ok) {
                // Server ist wieder verfügbar
                console.log(`Server ist wieder verfügbar nach ${attempt} Versuchen`);
                return;
            }
        } catch (error) {
            // Server noch nicht verfügbar - das ist normal während Neustart
            if (attempt % 5 === 0) { // Nur alle 5 Versuche loggen
                console.log(`Warte auf Server... (Versuch ${attempt}/${maxAttempts})`);
            }
        }
        
        // Warte vor nächstem Versuch
        await new Promise(resolve => setTimeout(resolve, delay));
    }
    
    // Wenn nach maxAttempts der Server noch nicht verfügbar ist, werfe Fehler
    throw new Error('Server ist nach dem Neustart nicht verfügbar. Bitte prüfen Sie die Server-Logs.');
}

