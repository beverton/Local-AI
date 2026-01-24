// API Base URL
const API_BASE = 'http://127.0.0.1:8000';

// State
let currentConversationId = null;
let currentModel = null;
let currentAbortController = null;
let currentImageAbortController = null;
let uploadedFilesList = []; // Liste der hochgeladenen Dateien
let settings = {
    temperature: 0.3,  // Niedriger für bessere Qualität (weniger "Jibberish")
    max_length: 2048,  // Default für Progress & Request (wird aus /performance/settings überschrieben)
    preferenceLearning: false,
    transcriptionLanguage: ""  // Leerer String = Auto-Erkennung
};

// DOM Elements
const conversationsList = document.getElementById('conversationsList');
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const btnSend = document.getElementById('btnSend');
const btnNewChat = document.getElementById('btnNewChat');
const btnNewImage = document.getElementById('btnNewImage');
// imageModelSelect entfernt - Modellauswahl erfolgt über Model-Service
const modelStatusText = document.getElementById('modelStatusText');
const modelStatusAudio = document.getElementById('modelStatusAudio');
const modelStatusImage = document.getElementById('modelStatusImage');
const chatTitle = document.getElementById('chatTitle');
const chatTitleInput = document.getElementById('chatTitleInput');
const statusText = document.getElementById('statusText');
const statusTextValue = document.getElementById('statusTextValue');
const statusImageValue = document.getElementById('statusImageValue');
const statusAudioValue = document.getElementById('statusAudioValue');
const statusTextModel = document.getElementById('statusTextModel');
const statusImageModel = document.getElementById('statusImageModel');
const statusAudioModel = document.getElementById('statusAudioModel');
const settingsPanel = document.getElementById('settingsPanel');
const btnSettings = document.getElementById('btnSettings');
const btnCloseSettings = document.getElementById('btnCloseSettings');
const btnRestart = document.getElementById('btnRestart');
const preferenceToggle = document.getElementById('preferenceToggle');
const btnResetPreferences = document.getElementById('btnResetPreferences');
const temperatureSlider = document.getElementById('temperatureSlider');
const temperatureValue = document.getElementById('temperatureValue');
const cpuPercent = document.getElementById('cpuPercent');
const ramPercent = document.getElementById('ramPercent');
const gpuPercent = document.getElementById('gpuPercent');
const btnCancel = document.getElementById('btnCancel');
const cpuThreadsSlider = document.getElementById('cpuThreadsSlider');
const cpuThreadsValue = document.getElementById('cpuThreadsValue');
const gpuOptimizationSelect = document.getElementById('gpuOptimizationSelect');
const disableCpuOffload = document.getElementById('disableCpuOffload');
const btnApplyPerformance = document.getElementById('btnApplyPerformance');
const primaryBudgetSlider = document.getElementById('primaryBudgetSlider');
const primaryBudgetValue = document.getElementById('primaryBudgetValue');
const gpuAllocationStatus = document.getElementById('gpuAllocationStatus');
const sidebarPrimaryBudget = document.getElementById('sidebarPrimaryBudget');
const sidebarSecondaryBudget = document.getElementById('sidebarSecondaryBudget');
const sidebarLoadedModels = document.getElementById('sidebarLoadedModels');
const gpuMaxPercent = document.getElementById('gpuMaxPercent');
const gpuPrimaryBudget = document.getElementById('gpuPrimaryBudget');
const gpuSecondaryBudget = document.getElementById('gpuSecondaryBudget');
const gpuLoadedModels = document.getElementById('gpuLoadedModels');
const chatInputContainer = document.getElementById('chatInputContainer');
const imageInputContainer = document.getElementById('imageInputContainer');
const imagePromptInput = document.getElementById('imagePromptInput');
const btnGenerateImage = document.getElementById('btnGenerateImage');
const btnCancelImage = document.getElementById('btnCancelImage');
const imageSizeModeRatio = document.getElementById('imageSizeModeRatio');
const imageSizeModeCustom = document.getElementById('imageSizeModeCustom');
const ratioModeContainer = document.getElementById('ratioModeContainer');
const customSizeModeContainer = document.getElementById('customSizeModeContainer');
const imageResolutionPreset = document.getElementById('imageResolutionPreset');
const imageAspectRatio = document.getElementById('imageAspectRatio');
const customRatioContainer = document.getElementById('customRatioContainer');
const customRatioW = document.getElementById('customRatioW');
const customRatioH = document.getElementById('customRatioH');
const ratioPreview = document.getElementById('ratioPreview');
const imageWidth = document.getElementById('imageWidth');
const imageHeight = document.getElementById('imageHeight');

// Auflösungs-Presets (werden vom Backend geladen)
let resolutionPresets = { s: 512, m: 720, l: 1024 };
const btnUpload = document.getElementById('btnUpload');
const fileInput = document.getElementById('fileInput');
const uploadedFiles = document.getElementById('uploadedFiles');
const fileUploadArea = document.getElementById('fileUploadArea');
const btnMicrophone = document.getElementById('btnMicrophone');
const btnAgentMode = document.getElementById('btnAgentMode');
const conversationModelSelect = document.getElementById('conversationModelSelect');
const btnModelManager = document.getElementById('btnModelManager');

// Helper-Funktion für Status-Updates
function updateModelStatus(modelType, status, modelId = null) {
    let statusElement, statusValueElement;
    
    switch(modelType) {
        case 'text':
            statusElement = statusTextModel;
            statusValueElement = statusTextValue;
            break;
        case 'image':
            statusElement = statusImageModel;
            statusValueElement = statusImageValue;
            break;
        case 'audio':
            statusElement = statusAudioModel;
            statusValueElement = statusAudioValue;
            break;
        default:
            return;
    }
    
    if (!statusElement || !statusValueElement) return;
    
    // Zeige Statusanzeige wenn Status nicht "Bereit" oder leer
    if (status && status !== 'Bereit' && status !== '-') {
        statusElement.style.display = 'flex';
        statusValueElement.textContent = status;
    } else if (modelId) {
        // Modell geladen - zeige Modell-ID
        statusElement.style.display = 'flex';
        statusValueElement.textContent = modelId || 'Bereit';
    } else {
        // Verstecke wenn kein Status
        statusElement.style.display = 'none';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // WICHTIG: setupEventListeners ZUERST aufrufen, damit Event-Listener registriert werden
    setupEventListeners();
    
    loadModels();
    // loadImageModels() entfernt - Modellauswahl erfolgt über Model-Service
    loadConversations();
    loadStatus();
    loadPreferences();
    loadPerformanceSettings();
    loadAudioSettings();
    loadQualitySettings();
    loadOutputSettings();
    startSystemStatsUpdate();
    
    // Lade initialen Status einmalig (kein automatisches Polling mehr)
    loadModelServiceStatus();
    
    // Debug: Prüfe ob Buttons gefunden wurden
    if (!btnMicrophone) {
        console.error('btnMicrophone nicht gefunden!');
    } else {
        console.log('✓ btnMicrophone gefunden und Event-Listener sollte registriert sein');
    }
    if (!btnUpload) {
        console.error('btnUpload nicht gefunden!');
    } else {
        console.log('✓ btnUpload gefunden und Event-Listener sollte registriert sein');
    }
    if (!fileInput) {
        console.error('fileInput nicht gefunden!');
    } else {
        console.log('✓ fileInput gefunden');
    }
});

// Event Listeners
// Quality Settings Funktionen (müssen vor setupEventListeners definiert sein)
async function loadQualitySettings() {
    try {
        const settings = await apiCall('/quality/settings');
        const qualityWebValidation = document.getElementById('qualityWebValidation');
        const qualityContradictionCheck = document.getElementById('qualityContradictionCheck');
        const qualityHallucinationCheck = document.getElementById('qualityHallucinationCheck');
        const qualityAutoWebSearch = document.getElementById('qualityAutoWebSearch');
        
        // Verwende explizite Defaults: false wenn nicht gesetzt (nicht true!)
        if (qualityWebValidation) qualityWebValidation.checked = settings.web_validation ?? false;
        if (qualityContradictionCheck) qualityContradictionCheck.checked = settings.contradiction_check ?? false;
        if (qualityHallucinationCheck) qualityHallucinationCheck.checked = settings.hallucination_check ?? false;
        if (qualityAutoWebSearch) qualityAutoWebSearch.checked = settings.auto_web_search ?? false;
    } catch (error) {
        console.error('Fehler beim Laden der Quality Settings:', error);
    }
}

async function saveQualitySettings() {
    try {
        const qualityWebValidation = document.getElementById('qualityWebValidation');
        const qualityContradictionCheck = document.getElementById('qualityContradictionCheck');
        const qualityHallucinationCheck = document.getElementById('qualityHallucinationCheck');
        const qualityAutoWebSearch = document.getElementById('qualityAutoWebSearch');
        
        await apiCall('/quality/settings', {
            method: 'POST',
            body: JSON.stringify({
                web_validation: qualityWebValidation ? qualityWebValidation.checked : false,
                contradiction_check: qualityContradictionCheck ? qualityContradictionCheck.checked : false,
                hallucination_check: qualityHallucinationCheck ? qualityHallucinationCheck.checked : false,
                auto_web_search: qualityAutoWebSearch ? qualityAutoWebSearch.checked : false
            })
        });
    } catch (error) {
        console.error('Fehler beim Speichern der Quality Settings:', error);
    }
}

function setupEventListeners() {
    btnSend.addEventListener('click', sendMessage);
    btnCancel.addEventListener('click', cancelGeneration);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    messageInput.addEventListener('input', () => {
        btnSend.disabled = messageInput.value.trim() === '';
        autoResizeTextarea();
    });
    btnNewChat.addEventListener('click', createNewConversation);
    btnNewImage.addEventListener('click', createNewImageConversation);
    btnSettings.addEventListener('click', () => {
        settingsPanel.classList.add('open');
    });
    btnCloseSettings.addEventListener('click', () => {
        settingsPanel.classList.remove('open');
    });
    if (btnRestart) {
        btnRestart.addEventListener('click', restartServer);
    }
    if (btnAgentMode) {
        btnAgentMode.addEventListener('click', toggleAgentMode);
    }
    preferenceToggle.addEventListener('change', togglePreferenceLearning);
    btnResetPreferences.addEventListener('click', resetPreferences);
    
    // Quality Settings Event Listeners
    const qualityWebValidation = document.getElementById('qualityWebValidation');
    const qualityContradictionCheck = document.getElementById('qualityContradictionCheck');
    const qualityHallucinationCheck = document.getElementById('qualityHallucinationCheck');
    const qualityAutoWebSearch = document.getElementById('qualityAutoWebSearch');
    
    if (qualityWebValidation) {
        qualityWebValidation.addEventListener('change', saveQualitySettings);
    }
    if (qualityContradictionCheck) {
        qualityContradictionCheck.addEventListener('change', saveQualitySettings);
    }
    if (qualityHallucinationCheck) {
        qualityHallucinationCheck.addEventListener('change', saveQualitySettings);
    }
    if (qualityAutoWebSearch) {
        qualityAutoWebSearch.addEventListener('change', saveQualitySettings);
    }
    temperatureSlider.addEventListener('input', (e) => {
        settings.temperature = parseFloat(e.target.value);
        temperatureValue.textContent = settings.temperature.toFixed(1);
    });
    cpuThreadsSlider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        cpuThreadsValue.textContent = value === 0 ? 'Auto' : value;
    });
    if (primaryBudgetSlider && primaryBudgetValue) {
        primaryBudgetSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            primaryBudgetValue.textContent = value === 0 ? 'Auto' : `${value}%`;
            // Update secondary budget display
            const maxPercent = parseFloat(gpuMaxPercent?.textContent || '90');
            const secondaryPercent = maxPercent - value;
            if (gpuSecondaryBudget) {
                gpuSecondaryBudget.textContent = secondaryPercent >= 0 ? `${secondaryPercent.toFixed(1)}%` : '0%';
            }
        });
    }
    btnApplyPerformance.addEventListener('click', applyPerformanceSettings);
    
    // Audio-Einstellungen
    const transcriptionLanguageSelect = document.getElementById('transcriptionLanguageSelect');
    if (transcriptionLanguageSelect) {
        transcriptionLanguageSelect.addEventListener('change', (e) => {
            settings.transcriptionLanguage = e.target.value || "";
            saveAudioSettings();
        });
    }
    
    // Output-Einstellungen
    const btnApplyOutputSettings = document.getElementById('btnApplyOutputSettings');
    const btnOpenOutputFolder = document.getElementById('btnOpenOutputFolder');
    if (btnApplyOutputSettings) {
        btnApplyOutputSettings.addEventListener('click', saveOutputSettings);
    }
    if (btnOpenOutputFolder) {
        btnOpenOutputFolder.addEventListener('click', openOutputFolder);
    }
    
    btnGenerateImage.addEventListener('click', generateImage);
    btnCancelImage.addEventListener('click', cancelImageGeneration);
    imagePromptInput.addEventListener('input', () => {
        btnGenerateImage.disabled = imagePromptInput.value.trim() === '';
        autoResizeImageTextarea();
    });
    
    // Größen-Modus Event-Listener
    if (imageSizeModeRatio && imageSizeModeCustom) {
        imageSizeModeRatio.addEventListener('change', handleSizeModeChange);
        imageSizeModeCustom.addEventListener('change', handleSizeModeChange);
    }
    
    // Ratio-Modus Event-Listener
    if (imageResolutionPreset) {
        imageResolutionPreset.addEventListener('change', updateRatioPreview);
    }
    if (imageAspectRatio) {
        imageAspectRatio.addEventListener('change', handleAspectRatioChange);
    }
    if (customRatioW && customRatioH) {
        customRatioW.addEventListener('input', handleCustomRatioChange);
        customRatioH.addEventListener('input', handleCustomRatioChange);
    }
    
    // Lade Preset-Werte vom Backend (optional - kann später implementiert werden)
    // loadResolutionPresets();
    
    // Initialisiere UI-Modus
    if (imageSizeModeRatio && imageSizeModeCustom) {
        handleSizeModeChange(); // Setze initialen Zustand
    }
    if (btnUpload && fileInput) {
        btnUpload.addEventListener('click', (e) => {
            console.log('btnUpload clicked!', e);
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            console.log('Triggering fileInput.click()');
            fileInput.click();
        }, true); // useCapture = true, damit Event früher gefangen wird
        fileInput.addEventListener('change', handleFileSelect);
        console.log('btnUpload Event-Listener registriert', btnUpload);
    } else {
        console.error('btnUpload oder fileInput nicht verfügbar:', {btnUpload, fileInput});
    }
    
    // Conversation Model Select (Settings)
    if (conversationModelSelect) {
        conversationModelSelect.addEventListener('change', async (e) => {
            const modelId = e.target.value || null;
            if (currentConversationId) {
                await setConversationModel(currentConversationId, modelId);
            }
        });
    }
    
    // Drag & Drop
    fileUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileUploadArea.style.background = 'var(--bg-primary)';
    });
    fileUploadArea.addEventListener('dragleave', () => {
        fileUploadArea.style.background = 'var(--bg-tertiary)';
    });
    fileUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        fileUploadArea.style.background = 'var(--bg-tertiary)';
        const files = Array.from(e.dataTransfer.files);
        handleFiles(files);
    });
    
    // Microphone recording
    if (btnMicrophone) {
        btnMicrophone.addEventListener('click', (e) => {
            console.log('btnMicrophone clicked!', e);
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            toggleMicrophone();
        }, true); // useCapture = true
        console.log('btnMicrophone Event-Listener registriert', btnMicrophone);
    } else {
        console.error('btnMicrophone nicht verfügbar');
    }
}

// switchTab Funktion entfernt - wird nicht mehr benötigt

// Image Size Mode Handler
function handleSizeModeChange() {
    if (!imageSizeModeRatio || !imageSizeModeCustom || !ratioModeContainer || !customSizeModeContainer) {
        return;
    }
    
    if (imageSizeModeRatio.checked) {
        // Ratio-Modus aktivieren
        ratioModeContainer.style.display = 'block';
        customSizeModeContainer.style.display = 'none';
    } else if (imageSizeModeCustom.checked) {
        // Custom Size-Modus aktivieren
        ratioModeContainer.style.display = 'none';
        customSizeModeContainer.style.display = 'block';
    }
}

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
}

function autoResizeImageTextarea() {
    imagePromptInput.style.height = 'auto';
    imagePromptInput.style.height = Math.min(imagePromptInput.scrollHeight, 200) + 'px';
}

// Image Size Ratio Functions
function updateRatioPreview() {
    if (!ratioPreview || !imageResolutionPreset || !imageAspectRatio) {
        return;
    }
    
    const preset = imageResolutionPreset.value;
    const ratio = imageAspectRatio.value;
    
    // Preset-Basis-Größe
    let baseSize = 1024;
    if (preset === 'm') {
        baseSize = 720;
    } else if (preset === 's') {
        baseSize = 512;
    }
    
    let width, height;
    
    if (ratio === 'custom' && customRatioW && customRatioH) {
        const w = parseFloat(customRatioW.value) || 1;
        const h = parseFloat(customRatioH.value) || 1;
        if (w > 0 && h > 0) {
            // Berechne Dimensionen basierend auf Preset und Custom Ratio
            const aspectRatio = w / h;
            if (aspectRatio >= 1) {
                // Breiter als hoch
                width = baseSize;
                height = Math.round(baseSize / aspectRatio);
            } else {
                // Höher als breit
                height = baseSize;
                width = Math.round(baseSize * aspectRatio);
            }
        } else {
            width = baseSize;
            height = baseSize;
        }
    } else {
        // Standard Aspect Ratios
        const ratioParts = ratio.split(':');
        if (ratioParts.length === 2) {
            const w = parseFloat(ratioParts[0]);
            const h = parseFloat(ratioParts[1]);
            if (w > 0 && h > 0) {
                const aspectRatio = w / h;
                if (aspectRatio >= 1) {
                    width = baseSize;
                    height = Math.round(baseSize / aspectRatio);
                } else {
                    height = baseSize;
                    width = Math.round(baseSize * aspectRatio);
                }
            } else {
                width = baseSize;
                height = baseSize;
            }
        } else {
            width = baseSize;
            height = baseSize;
        }
    }
    
    ratioPreview.textContent = `Dimensionen: ${width} x ${height} px`;
}

function handleAspectRatioChange() {
    if (!imageAspectRatio || !customRatioContainer) {
        return;
    }
    
    const ratio = imageAspectRatio.value;
    if (ratio === 'custom') {
        customRatioContainer.style.display = 'block';
    } else {
        customRatioContainer.style.display = 'none';
    }
    
    updateRatioPreview();
}

function handleCustomRatioChange() {
    updateRatioPreview();
}

// API Functions
async function apiCall(endpoint, options = {}) {
    try {
        const fetchOptions = {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        };
        
        // Preserve signal if provided (for cancellation)
        if (options.signal) {
            fetchOptions.signal = options.signal;
        }
        
        // Debug-Logging entfernt (Port 7244 Service nicht verfügbar)
        const response = await fetch(`${API_BASE}${endpoint}`, fetchOptions);
        
        if (!response.ok) {
            // Prüfe ob es ein "model_loading" Status ist (202 Accepted)
            if (response.status === 202) {
                const errorData = await response.json();
                // Debug-Logging entfernt
                if (errorData.detail && errorData.detail.status === "model_loading") {
                    throw { name: "ModelLoading", detail: errorData.detail };
                }
            }
            const error = await response.json();
            throw new Error(error.detail || 'API Fehler');
        }
        
        return await response.json();
    } catch (error) {
        // Debug-Logging entfernt
        console.error('API Error:', error);
        
        // Behandle spezifische Fehlertypen
        // AbortError = Request wurde abgebrochen (normal, nicht als Netzwerkfehler behandeln)
        if (error.name === 'AbortError') {
            throw error; // Weiterwerfen als AbortError
        }
        
        // TypeError mit "Failed to fetch" = echter Netzwerkfehler
        // NetworkError = echter Netzwerkfehler
        // CONNECTION_REFUSED = Server nicht erreichbar
        const isRealNetworkError = (
            (error instanceof TypeError && error.message && error.message.includes('Failed to fetch')) ||
            error.name === 'NetworkError' ||
            (error.message && (error.message.includes('CONNECTION_REFUSED') || error.message.includes('Network request failed')))
        );
        
        if (isRealNetworkError) {
            throw new Error('Netzwerkfehler: Server nicht erreichbar. Bitte prüfen Sie, ob der Server läuft.');
        }
        
        // Alle anderen Fehler weiterwerfen (inkl. ModelLoading, etc.)
        throw error;
    }
}

async function waitForModelLoad(modelType, modelId, conversationId = null) {
    /**
     * Wartet bis ein Modell geladen ist.
     * Pollt den Status-Endpoint bis Modell fertig ist.
     * 
     * @param {string} modelType - "text", "image" oder "audio"
     * @param {string} modelId - Die ID des Modells
     * @param {string|null} conversationId - Optional - ID der Conversation die das Modell benötigt
     * @returns {Promise<boolean>} True wenn Modell geladen ist
     */
    const statusEndpoint = modelType === "text" ? "/models/load/status" : 
                          modelType === "image" ? "/image/models/load/status" : 
                          "/audio/models/load/status";
    const maxAttempts = 24; // Max 2 Minuten (5 Sekunden pro Poll)
    let attempts = 0;
    
    // Erstelle Fortschrittsanzeige im Chat-Feld
    let progressMessageId = null;
    let progressBar = null;
    let progressText = null;
    
    if (conversationId) {
        // Zeige Fortschrittsanzeige nur wenn wir in einer Conversation sind
        progressMessageId = addMessageToChat('assistant', `Lädt Modell ${modelId}...`, true);
        const progressMsg = document.getElementById(progressMessageId);
        if (progressMsg) {
            const contentDiv = progressMsg.querySelector('.message-content');
            if (contentDiv) {
                contentDiv.innerHTML = `
                    <div>Lädt Modell ${modelId}...</div>
                    <div class="model-loading-progress">
                        <div class="progress-bar-container">
                            <div class="progress-bar" id="model-progress-bar-${progressMessageId}"></div>
                        </div>
                        <div class="progress-text" id="model-progress-text-${progressMessageId}">0%</div>
                    </div>
                `;
                progressBar = document.getElementById(`model-progress-bar-${progressMessageId}`);
                progressText = document.getElementById(`model-progress-text-${progressMessageId}`);
            }
        }
    }
    
    const startTime = Date.now();
    const estimatedDuration = 60000; // Geschätzte Dauer: 60 Sekunden
    
    while (attempts < maxAttempts) {
        try {
            const status = await apiCall(statusEndpoint);
            
            // Update Fortschrittsanzeige basierend auf Zeit
            if (progressBar && progressText) {
                const elapsed = Date.now() - startTime;
                const progress = Math.min((elapsed / estimatedDuration) * 100, 95); // Max 95% bis fertig
                progressBar.style.width = `${progress}%`;
                progressText.textContent = `${Math.round(progress)}%`;
            }
            
            if (!status.loading) {
                // Modell-Laden abgeschlossen
                if (status.error) {
                    // Entferne Fortschrittsanzeige bei Fehler
                    if (progressMessageId) {
                        const progressMsg = document.getElementById(progressMessageId);
                        if (progressMsg) {
                            progressMsg.remove();
                        }
                    }
                    throw new Error(status.error);
                }
                if (status.model_id === modelId) {
                    // Entferne Fortschrittsanzeige bei Erfolg
                    if (progressMessageId) {
                        const progressMsg = document.getElementById(progressMessageId);
                        if (progressMsg) {
                            progressMsg.remove();
                        }
                    }
                    return true; // Modell erfolgreich geladen
                }
            }
            
            // Warte 500ms bevor nächster Poll (für schnellere Updates)
            await new Promise(resolve => setTimeout(resolve, 500));
            attempts++;
        } catch (error) {
            // Bei Netzwerkfehlern (Server-Neustart), versuche es nochmal ohne zu loggen
            if (error.message && (error.message.includes('fetch') || error.message.includes('Netzwerkfehler') || error.message.includes('CONNECTION_REFUSED'))) {
                // Server könnte neu gestartet haben - warte länger und versuche es nochmal
                if (attempts < maxAttempts - 1) { // Nur wenn noch Versuche übrig sind
                    attempts++;
                    await new Promise(resolve => setTimeout(resolve, 3000)); // Warte länger bei Netzwerkfehlern
                    continue;
                }
            }
            // Entferne Fortschrittsanzeige bei Fehler
            if (progressMessageId) {
                const progressMsg = document.getElementById(progressMessageId);
                if (progressMsg) {
                    progressMsg.remove();
                }
            }
            // Nur andere Fehler loggen
            if (!error.message || (!error.message.includes('fetch') && !error.message.includes('Netzwerkfehler') && !error.message.includes('CONNECTION_REFUSED'))) {
                console.error('Fehler beim Prüfen des Modell-Status:', error);
            }
            throw error;
        }
    }
    
    // Entferne Fortschrittsanzeige bei Timeout
    if (progressMessageId) {
        const progressMsg = document.getElementById(progressMessageId);
        if (progressMsg) {
            progressMsg.remove();
        }
    }
    
    throw new Error("Timeout beim Warten auf Modell-Laden (mehr als 2 Minuten)");
}

// Lade Model-Service-Status
async function loadModelServiceStatus() {
    try {
        const response = await fetch('/model-service/status');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const status = await response.json();
        
        // Update Text-Modell Status
        if (status.text_model) {
            const textStatus = status.text_model;
            if (textStatus.loaded) {
                modelStatusText.textContent = textStatus.model_id || 'Geladen';
                modelStatusText.style.color = 'var(--success)';
            } else if (textStatus.loading) {
                modelStatusText.textContent = 'Lädt...';
                modelStatusText.style.color = 'var(--warning)';
            } else {
                modelStatusText.textContent = 'Nicht geladen';
                modelStatusText.style.color = 'var(--text-secondary)';
            }
        }
        
        // Update Audio-Modell Status
        if (status.audio_model) {
            const audioStatus = status.audio_model;
            if (audioStatus.loaded) {
                modelStatusAudio.textContent = audioStatus.model_id || 'Geladen';
                modelStatusAudio.style.color = 'var(--success)';
            } else if (audioStatus.loading) {
                modelStatusAudio.textContent = 'Lädt...';
                modelStatusAudio.style.color = 'var(--warning)';
            } else {
                modelStatusAudio.textContent = 'Nicht geladen';
                modelStatusAudio.style.color = 'var(--text-secondary)';
            }
        }
        
        // Update Image-Modell Status
        if (status.image_model) {
            const imageStatus = status.image_model;
            if (imageStatus.loaded) {
                modelStatusImage.textContent = imageStatus.model_id || 'Geladen';
                modelStatusImage.style.color = 'var(--success)';
            } else if (imageStatus.loading) {
                modelStatusImage.textContent = 'Lädt...';
                modelStatusImage.style.color = 'var(--warning)';
            } else {
                modelStatusImage.textContent = 'Nicht geladen';
                modelStatusImage.style.color = 'var(--text-secondary)';
            }
        }
        
        // Update GPU Allocation Display
        if (status.gpu_allocation) {
            const allocation = status.gpu_allocation;
            
            // Update settings panel display
            if (gpuPrimaryBudget) {
                gpuPrimaryBudget.textContent = allocation.primary_budget_percent !== null && allocation.primary_budget_percent !== undefined 
                    ? `${allocation.primary_budget_percent.toFixed(1)}%` : 'Auto';
            }
            if (gpuSecondaryBudget) {
                gpuSecondaryBudget.textContent = allocation.secondary_budget_percent !== null && allocation.secondary_budget_percent !== undefined 
                    ? `${allocation.secondary_budget_percent.toFixed(1)}%` : '-';
            }
            
            // Update loaded models display
            const loadedModels = [];
            if (allocation.primary_models && allocation.primary_models.length > 0) {
                loadedModels.push(`Primary: ${allocation.primary_models.join(', ')}`);
            }
            if (allocation.secondary_models && allocation.secondary_models.length > 0) {
                loadedModels.push(`Secondary: ${allocation.secondary_models.join(', ')}`);
            }
            if (gpuLoadedModels) {
                gpuLoadedModels.textContent = loadedModels.length > 0 ? loadedModels.join(' | ') : 'Keine Modelle geladen';
            }
            
            // Update sidebar display
            if (gpuAllocationStatus) {
                if (allocation.primary_models && allocation.primary_models.length > 0 || 
                    allocation.secondary_models && allocation.secondary_models.length > 0) {
                    gpuAllocationStatus.style.display = 'block';
                    
                    if (sidebarPrimaryBudget) {
                        sidebarPrimaryBudget.textContent = allocation.primary_budget_percent !== null && allocation.primary_budget_percent !== undefined 
                            ? `${allocation.primary_budget_percent.toFixed(1)}%` : 'Auto';
                    }
                    if (sidebarSecondaryBudget) {
                        sidebarSecondaryBudget.textContent = allocation.secondary_budget_percent !== null && allocation.secondary_budget_percent !== undefined 
                            ? `${allocation.secondary_budget_percent.toFixed(1)}%` : '-';
                    }
                    if (sidebarLoadedModels) {
                        const sidebarModels = [];
                        if (allocation.primary_models && allocation.primary_models.length > 0) {
                            sidebarModels.push(allocation.primary_models.map(m => m === 'text' ? 'Text' : m === 'image' ? 'Bild' : m).join(', '));
                        }
                        if (allocation.secondary_models && allocation.secondary_models.length > 0) {
                            sidebarModels.push(allocation.secondary_models.map(m => m === 'audio' ? 'Audio' : m).join(', '));
                        }
                        sidebarLoadedModels.textContent = sidebarModels.length > 0 ? sidebarModels.join(' | ') : '-';
                    }
                } else {
                    gpuAllocationStatus.style.display = 'none';
                }
            }
        }
    } catch (error) {
        console.error('Fehler beim Laden des Model-Service-Status:', error);
        // Zeige Fehler-Status
        if (modelStatusText) modelStatusText.textContent = 'Fehler';
        if (modelStatusAudio) modelStatusAudio.textContent = 'Fehler';
        if (modelStatusImage) modelStatusImage.textContent = 'Fehler';
    }
}

async function loadModels() {
    // Diese Funktion wird nicht mehr benötigt, aber für Conversation-Modell-Auswahl behalten
    try {
        // Lade Modelle nur für Conversation-Modell-Auswahl
        await loadConversationModels();
    } catch (error) {
        console.error('Fehler beim Laden der Modelle:', error);
    }
}

async function loadConversationModels() {
    try {
        const data = await apiCall('/models');
        const optionsHtml = '<option value="">Globales Modell verwenden</option>';
        
        // Nur Text-Modelle (keine Image- oder Audio-Modelle)
        let options = '';
        for (const [id, model] of Object.entries(data.models)) {
            if (model.type !== 'image' && model.type !== 'audio') {
                options += `<option value="${id}">${escapeHtml(model.name)}</option>`;
            }
        }
        
        // Update both selectors
        if (conversationModelSelect) {
            conversationModelSelect.innerHTML = optionsHtml + options;
        }
    } catch (error) {
        console.error('Fehler beim Laden der Conversation-Modelle:', error);
    }
}

// loadImageModels() entfernt - Modellauswahl erfolgt automatisch über Model-Service
// loadImageModel() entfernt - Modellauswahl erfolgt automatisch über Model-Service
// pollImageModelLoadStatus() entfernt - nicht mehr benötigt

// loadModel Funktion entfernt - Modelle werden jetzt über Model-Service verwaltet
// Benutzer sollten den Model Manager verwenden, um Modelle zu laden

// pollModelLoadStatus Funktion entfernt - Status wird jetzt über Model-Service-Status angezeigt

async function loadStatus() {
    try {
        const status = await apiCall('/status');
        if (status.model_loaded) {
            updateModelStatus('text', null, status.current_model);
        } else {
            updateModelStatus('text', 'Kein Modell geladen');
        }
    } catch (error) {
        console.error('Fehler beim Laden des Status:', error);
    }
}

async function loadConversations() {
    try {
        const data = await apiCall('/conversations');
        // Null-Check: Falls keine Conversations vorhanden sind
        const conversations = data?.conversations || [];
        renderConversations(conversations);
    } catch (error) {
        console.error('Fehler beim Laden der Gespräche:', error);
        conversationsList.innerHTML = '<div class="loading">Fehler beim Laden</div>';
    }
}

function renderConversations(conversations) {
    if (conversations.length === 0) {
        conversationsList.innerHTML = '<div class="loading">Keine Gespräche</div>';
        return;
    }
    
    // Speichere aktuelle Scroll-Position
    const scrollTop = conversationsList.scrollTop;
    
    conversationsList.innerHTML = conversations.map(conv => {
        const modelBadge = conv.model_id ? `<span class="conversation-model-badge">${escapeHtml(conv.model_id)}</span>` : '';
        const conversationType = conv.conversation_type || "chat";
        const typeIcon = conversationType === "image" ? "🖼️ " : "";
        return `
        <div class="conversation-item ${conv.id === currentConversationId ? 'active' : ''} ${conversationType === 'image' ? 'conversation-type-image' : ''}" 
             data-id="${conv.id}">
            <input type="checkbox" class="conversation-checkbox" data-id="${conv.id}" style="margin-right: 8px; cursor: pointer;">
            <div class="conversation-content">
                <span class="conversation-title">${typeIcon}${escapeHtml(conv.title)}</span>
                ${modelBadge}
            </div>
            <button class="conversation-delete" onclick="deleteConversation('${conv.id}', event)">×</button>
        </div>
    `;
    }).join('');
    
    // Restore scroll position
    conversationsList.scrollTop = scrollTop;
    
    
    // Checkbox-Event-Handler für Mehrfachauswahl (vor Item-Click-Handler)
    conversationsList.querySelectorAll('.conversation-checkbox').forEach(checkbox => {
        checkbox.addEventListener('click', (e) => {
            e.stopPropagation();
        });
        checkbox.addEventListener('change', (e) => {
            e.stopPropagation();
            updateSelectionUI();
        });
    });
    
    // Add click listeners
    conversationsList.querySelectorAll('.conversation-item').forEach(item => {
        item.addEventListener('click', (e) => {
            // Ignoriere Klicks auf Checkbox und Delete-Button
            if (e.target.classList.contains('conversation-delete') || e.target.classList.contains('conversation-checkbox') || e.target.tagName === 'INPUT') {
                return;
            }
            // Erlaube Wechsel auch während laufender Operationen
            loadConversation(item.dataset.id);
        });
    });
    
    // Initialisiere Selection-UI
    updateSelectionUI();
}

async function loadConversation(conversationId) {
    try {
        const conversation = await apiCall(`/conversations/${conversationId}`);
        currentConversationId = conversationId;
        updateChatTitle(conversation.title);
        
        // Clear and render messages
        chatMessages.innerHTML = '';
        conversation.messages.forEach(msg => {
            // Prüfe ob Message ein Bild ist
            if (msg.content === "image" && msg.image_base64) {
                addImageToChat(msg.prompt || "Generiertes Bild", `data:image/png;base64,${msg.image_base64}`);
            } else {
                addMessageToChat(msg.role, msg.content);
            }
        });
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Update active conversation
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.toggle('active', item.dataset.id === conversationId);
        });
        
        // Zeige entsprechendes Interface basierend auf conversation_type
        const conversationType = conversation.conversation_type || "chat";
        if (conversationType === "image") {
            chatInputContainer.style.display = 'none';
            imageInputContainer.style.display = 'block';
        } else {
            chatInputContainer.style.display = 'block';
            imageInputContainer.style.display = 'none';
            // Reset Button-Status beim Wechseln zu einer Chat-Conversation
            btnSend.style.display = 'block';
            btnCancel.style.display = 'none';
            btnSend.disabled = messageInput.value.trim() === '';
            messageInput.disabled = false;
        }
        
        // Lade Conversation-Modell
        const modelId = conversation.model_id || '';
        if (conversationModelSelect) {
            conversationModelSelect.value = modelId;
        }
        
        // Lade Agent-Modus Status
        if (conversationType === "chat") {
            await loadAgentMode();
        } else {
            // Agent-Modus nur für Chat-Conversations
            updateAgentModeButton(false);
        }
        
        // Setze AbortController zurück, wenn zu anderer Conversation gewechselt wurde
        if (currentAbortController) {
            currentAbortController = null;
        }
        
        loadConversations(); // Refresh list
    } catch (error) {
        console.error('Fehler beim Laden der Conversation:', error);
        alert('Fehler beim Laden der Conversation: ' + error.message);
    }
}

async function setConversationModel(conversationId, modelId) {
    try {
        await apiCall(`/conversations/${conversationId}/model`, {
            method: 'POST',
            body: JSON.stringify({ model_id: modelId })
        });
        console.log(`Modell für Conversation ${conversationId} gesetzt: ${modelId || 'Global'}`);
    } catch (error) {
        console.error('Fehler beim Setzen des Conversation-Modells:', error);
        alert('Fehler beim Setzen des Modells: ' + error.message);
        // Revert selection
        const conversation = await apiCall(`/conversations/${conversationId}`);
        const modelId = conversation.model_id || '';
        if (conversationModelSelect) {
            conversationModelSelect.value = modelId;
        }
    }
}

function updateChatTitle(title) {
    chatTitle.textContent = title;
    if (chatTitleInput) {
        chatTitleInput.value = title;
    }
}

function enableTitleEditing() {
    if (!chatTitle || !chatTitleInput || !currentConversationId) return;
    
    chatTitle.style.display = 'none';
    chatTitleInput.style.display = 'block';
    chatTitleInput.focus();
    chatTitleInput.select();
    
    const saveTitle = async () => {
        const newTitle = chatTitleInput.value.trim();
        if (newTitle && newTitle !== chatTitle.textContent) {
            try {
                await apiCall(`/conversations/${currentConversationId}/title`, {
                    method: 'PATCH',
                    body: JSON.stringify({ title: newTitle })
                });
                updateChatTitle(newTitle);
                loadConversations(); // Aktualisiere Liste
            } catch (error) {
                console.error('Fehler beim Aktualisieren des Titels:', error);
                chatTitleInput.value = chatTitle.textContent;
            }
        } else {
            chatTitleInput.value = chatTitle.textContent;
        }
        chatTitle.style.display = 'block';
        chatTitleInput.style.display = 'none';
    };
    
    chatTitleInput.onblur = saveTitle;
    chatTitleInput.onkeydown = (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            chatTitleInput.blur();
        } else if (e.key === 'Escape') {
            chatTitleInput.value = chatTitle.textContent;
            chatTitle.style.display = 'block';
            chatTitleInput.style.display = 'none';
        }
    };
}

// Doppelklick auf Titel zum Editieren
if (chatTitle) {
    chatTitle.addEventListener('dblclick', enableTitleEditing);
}

function updateSelectionUI() {
    const selected = getSelectedConversationIds();
    const items = conversationsList.querySelectorAll('.conversation-item');
    items.forEach(item => {
        const checkbox = item.querySelector('.conversation-checkbox');
        if (checkbox && selected.includes(checkbox.dataset.id)) {
            item.classList.add('selected');
        } else {
            item.classList.remove('selected');
        }
    });
}

function getSelectedConversationIds() {
    const checkboxes = conversationsList.querySelectorAll('.conversation-checkbox:checked');
    return Array.from(checkboxes).map(cb => cb.dataset.id);
}

async function deleteSelectedConversations() {
    const selected = getSelectedConversationIds();
    if (selected.length === 0) return;
    
    if (!confirm(`Wirklich ${selected.length} Gespräch${selected.length > 1 ? 'e' : ''} löschen?`)) {
        return;
    }
    
    try {
        await apiCall('/conversations/delete-multiple', {
            method: 'POST',
            body: JSON.stringify({ conversation_ids: selected })
        });
        
        if (selected.includes(currentConversationId)) {
            createNewConversation();
        }
        loadConversations();
    } catch (error) {
        console.error('Fehler beim Löschen:', error);
        alert('Fehler beim Löschen: ' + error.message);
    }
}

// Entf-Taste-Handler
document.addEventListener('keydown', (e) => {
    if (e.key === 'Delete' || e.key === 'Backspace') {
        const selected = getSelectedConversationIds();
        if (selected.length > 0 && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
            e.preventDefault();
            deleteSelectedConversations();
        }
    }
});

async function createNewConversation() {
    try {
        // Setze UI sofort (optimistic update)
        updateChatTitle('Neues Gespräch');
        chatMessages.innerHTML = '<div class="welcome-message"><h3>Neues Gespräch</h3><p>Stellen Sie eine Frage oder starten Sie ein Gespräch.</p></div>';
        
        // Zeige Chat-Interface
        chatInputContainer.style.display = 'block';
        imageInputContainer.style.display = 'none';
        
        // Reset Conversation-Modell-Auswahl
        if (conversationModelSelect) {
            conversationModelSelect.value = '';
        }
        
        // Reset UI-Elemente
        messageInput.disabled = false;
        btnSend.disabled = messageInput.value.trim() === '';
        btnSend.style.display = 'block';
        btnCancel.style.display = 'none';
        
        // Erstelle Conversation im Hintergrund
        const data = await apiCall('/conversations', { method: 'POST' });
        currentConversationId = data.conversation_id;
        
        loadConversations();
    } catch (error) {
        console.error('Fehler beim Erstellen der Conversation:', error);
        alert('Fehler beim Erstellen der Conversation: ' + error.message);
    }
}

async function createNewImageConversation() {
    try {
        // Setze UI sofort (optimistic update)
        updateChatTitle('Neues Bild');
        chatMessages.innerHTML = '<div class="welcome-message"><h3>Neues Bild</h3><p>Beschreiben Sie das Bild, das Sie generieren möchten.</p></div>';
        
        // Zeige Image-Interface, verstecke Chat-Interface
        chatInputContainer.style.display = 'none';
        imageInputContainer.style.display = 'block';
        
        // Reset Image-Input
        imagePromptInput.value = '';
        imagePromptInput.disabled = false;
        btnGenerateImage.disabled = true;
        btnGenerateImage.style.display = 'block';
        btnCancelImage.style.display = 'none';
        
        // Erstelle Image-Conversation im Hintergrund
        const data = await apiCall('/conversations/image', { method: 'POST' });
        currentConversationId = data.conversation_id;
        
        loadConversations();
    } catch (error) {
        console.error('Fehler beim Erstellen der Bild-Conversation:', error);
        alert('Fehler beim Erstellen der Bild-Conversation: ' + error.message);
    }
}

async function deleteConversation(conversationId, event) {
    event.stopPropagation();
    if (!confirm('Gespräch wirklich löschen?')) return;
    
    
    try {
        // Lösche sofort aus UI (optimistic update)
        const item = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
        if (item) {
            item.style.opacity = '0.5';
            item.style.pointerEvents = 'none';
        }
        
        await apiCall(`/conversations/${conversationId}`, { method: 'DELETE' });
        
        if (conversationId === currentConversationId) {
            createNewConversation();
        }
        loadConversations();
    } catch (error) {
        console.error('Fehler beim Löschen:', error);
        alert('Fehler beim Löschen: ' + error.message);
        // Restore UI
        const item = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
        if (item) {
            item.style.opacity = '1';
            item.style.pointerEvents = 'auto';
        }
        loadConversations();
    }
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message && uploadedFilesList.length === 0) return;
    
    // Prüfe ob wir in einer chat-Conversation sind
    if (currentConversationId) {
        const conversation = await apiCall(`/conversations/${currentConversationId}`);
        const conversationType = conversation.conversation_type || "chat";
        if (conversationType !== "chat") {
            alert('Diese Funktion ist nur für Chat-Conversations verfügbar!');
            return;
        }
    }
    
    // Speichere aktuelle Conversation-ID (kann sich während der Operation ändern)
    const conversationIdAtStart = currentConversationId;
    
    // Prepare message with file context
    let fullMessage = message;
    if (uploadedFilesList.length > 0) {
        const fileContexts = uploadedFilesList.map(f => 
            `[Datei: ${f.name}]\n${f.preview || f.content}`
        ).join('\n\n');
        fullMessage = fileContexts + (message ? '\n\n' + message : '');
    }
    
    // Add user message to chat (with file previews)
    const messageDisplay = message || `Datei${uploadedFilesList.length > 1 ? 'en' : ''} hochgeladen: ${uploadedFilesList.map(f => f.name).join(', ')}`;
    addMessageToChat('user', messageDisplay, false, uploadedFilesList);
    messageInput.value = '';
    btnSend.disabled = true;
    messageInput.disabled = true;
    autoResizeTextarea();
    
    // Clear uploaded files
    uploadedFilesList = [];
    renderUploadedFiles();
    
    // Show cancel button, hide send button
    btnSend.style.display = 'none';
    btnCancel.style.display = 'block';
    
    // Create AbortController for cancellation
    currentAbortController = new AbortController();
    
    // Show loading indicator
    const loadingId = addMessageToChat('assistant', 'Denkt nach...', true);
    
    try {
        let response;
        let retries = 0;
        const maxRetries = 3;
        
        // Streaming als Standard (SSE über /chat/stream).
        // Wenn Streaming aus irgendeinem Grund fehlschlägt, fallback auf /chat (non-stream).
        try {
            await sendMessageStream(fullMessage, conversationIdAtStart, loadingId);
            return; // Streaming beendet, keine weitere Verarbeitung nötig
        } catch (error) {
            console.warn('Streaming fehlgeschlagen, verwende normale Methode:', error);
            // Weiter mit Fallback unten
        }
        
        while (retries < maxRetries) {
            try {
                response = await apiCall('/chat', {
                    method: 'POST',
                    body: JSON.stringify({
                        message: fullMessage,
                        conversation_id: conversationIdAtStart,
                        temperature: settings.temperature
                    }),
                    signal: currentAbortController.signal
                });
                break; // Erfolgreich - verlasse Schleife
            } catch (error) {
                // Prüfe ob es ein ModelLoading-Fehler ist
                if (error.name === "ModelLoading" || (error.detail && error.detail.status === "model_loading")) {
                    const modelId = error.detail?.model_id || error.detail?.detail?.model_id;
                    if (modelId) {
                        // Zeige Lade-Status
                        const loadingMsg = document.getElementById(loadingId);
                        if (loadingMsg) {
                            loadingMsg.querySelector('.message-content').textContent = `Lade Modell ${modelId}...`;
                        }
                        
                        // Warte auf Modell-Laden
                        await waitForModelLoad("text", modelId, conversationIdAtStart);
                        
                        // Wiederhole Request
                        retries++;
                        continue;
                    }
                }
                // Anderer Fehler - weiterwerfen
                throw error;
            }
        }
        
        if (!response) {
            throw new Error("Maximale Anzahl von Wiederholungen erreicht");
        }
        
        // Prüfe ob User zu einer anderen Conversation gewechselt hat
        if (currentConversationId !== conversationIdAtStart) {
            // User hat gewechselt - füge Nachricht zur ursprünglichen Conversation hinzu
            // (wird beim nächsten Laden angezeigt)
            console.log('User hat zu anderer Conversation gewechselt während der Generierung');
            return;
        }
        
        // Update conversation ID if new
        if (response.conversation_id !== currentConversationId) {
            currentConversationId = response.conversation_id;
            loadConversations();
        }
        
        // Remove loading message and add real response
        const loadingMsg = document.getElementById(loadingId);
        if (loadingMsg) {
            loadingMsg.remove();
        }
        
        // Debug: Prüfe Response-Format
        console.log('Response received:', response);
        console.log('Response.response:', response.response);
        console.log('Response type:', typeof response);
        
        if (!response || !response.response) {
            console.error('Response ist leer oder hat kein response-Feld!', response);
            addMessageToChat('assistant', 'Fehler: Keine Antwort erhalten');
            return;
        }
        
        addMessageToChat('assistant', response.response);
        
        // Update title if first message
        if (chatTitle.textContent === 'Neues Gespräch') {
            updateChatTitle(message.substring(0, 50) + (message.length > 50 ? '...' : ''));
        }
        
    } catch (error) {
        // Prüfe ob User zu einer anderen Conversation gewechselt hat
        if (currentConversationId !== conversationIdAtStart) {
            console.log('User hat zu anderer Conversation gewechselt während der Generierung');
            // Button trotzdem zurücksetzen, auch wenn Conversation gewechselt wurde
            btnSend.style.display = 'block';
            btnCancel.style.display = 'none';
            btnSend.disabled = messageInput.value.trim() === '';
            messageInput.disabled = false;
            currentAbortController = null;
            return;
        }
        
        // Check if it was aborted
        if (error.name === 'AbortError') {
            const loadingMsg = document.getElementById(loadingId);
            if (loadingMsg) {
                loadingMsg.remove();
            }
            addMessageToChat('assistant', 'Generierung abgebrochen.');
        } else {
            // Remove loading message
            const loadingMsg = document.getElementById(loadingId);
            if (loadingMsg) {
                loadingMsg.remove();
            }
            addMessageToChat('assistant', 'Fehler: ' + error.message);
            console.error('Fehler beim Senden:', error);
        }
    } finally {
        // Reset UI - IMMER ausführen, auch bei frühen Returns
        // Setze Button IMMER zurück, wenn wir noch in derselben Conversation sind
        // Wenn User zu anderer Conversation gewechselt hat, wird Button in loadConversation zurückgesetzt
        const userSwitchedConversation = conversationIdAtStart !== null && 
                                         currentConversationId !== null && 
                                         currentConversationId !== conversationIdAtStart;
        
        if (!userSwitchedConversation) {
            // User hat nicht gewechselt - setze Button zurück
            btnSend.style.display = 'block';
            btnCancel.style.display = 'none';
            btnSend.disabled = messageInput.value.trim() === '';
            messageInput.disabled = false;
            currentAbortController = null;
        } else {
            // User hat gewechselt - Button wird in loadConversation zurückgesetzt
            // Setze nur AbortController zurück
            currentAbortController = null;
        }
    }
}

function cancelGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
        updateModelStatus('text', 'Abgebrochen');
        // Button wird im finally Block von sendMessage zurückgesetzt
        // Der AbortController wird im finally Block auf null gesetzt
    }
}

async function generateImage() {
    const prompt = imagePromptInput.value.trim();
    if (!prompt) return;
    
    // Prüfe ob wir in einer image-Conversation sind
    if (!currentConversationId) {
        alert('Bitte erstellen Sie zuerst eine Bild-Conversation!');
        return;
    }
    
    // Modellauswahl erfolgt automatisch über Model-Service
    // Keine manuelle Modellauswahl mehr nötig
    
    // Disable input
    imagePromptInput.disabled = true;
    btnGenerateImage.style.display = 'none';
    btnCancelImage.style.display = 'block';
    
    // Create AbortController
    currentImageAbortController = new AbortController();
    
    // Show loading message
    const loadingId = addMessageToChat('assistant', 'Generiere Bild...', true);
    updateModelStatus('image', 'Generiere Bild...');
    
    // Deklariere Variablen einmal am Anfang
    let loadingMsg = document.getElementById(loadingId);
    let progressBar = null;
    let progressText = null;
    let progressInterval = null;
    
    try {
        let response;
        let retries = 0;
        const maxRetries = 3;
        
        // Entweder-Oder: Ratio ODER Custom Size
        const requestBody = {
            prompt: prompt,
            negative_prompt: "",
            num_inference_steps: 20,
            guidance_scale: 7.5,
            // model_id wird automatisch vom Model-Service verwendet
            conversation_id: currentConversationId
        };
        
        if (imageSizeModeRatio && imageSizeModeRatio.checked) {
            // Ratio-Modus: verwende aspect_ratio mit Preset-Präfix
            const preset = imageResolutionPreset.value;
            const ratio = imageAspectRatio.value;
            
            if (ratio === 'custom' && customRatioW && customRatioH) {
                const w = parseFloat(customRatioW.value);
                const h = parseFloat(customRatioH.value);
                if (w && h && w > 0 && h > 0) {
                    // Format: "preset:custom:W:H"
                    requestBody.aspect_ratio = `${preset}:custom:${w}:${h}`;
                }
            } else if (ratio) {
                // Format: "preset:ratio" z.B. "l:16:9"
                requestBody.aspect_ratio = `${preset}:${ratio}`;
            }
            
            // Backend wird Dimensionen aus Ratio + Preset berechnen
        } else {
            // Custom Size-Modus: verwende width/height
            const width = parseInt(imageWidth.value) || 1024;
            const height = parseInt(imageHeight.value) || 1024;
            requestBody.width = width;
            requestBody.height = height;
        }
        
        // Erstelle Fortschrittsanzeige (nach requestBody Definition)
        loadingMsg = document.getElementById(loadingId);
        if (loadingMsg) {
            const contentDiv = loadingMsg.querySelector('.message-content');
            if (contentDiv) {
                contentDiv.innerHTML = `
                    <div>Generiere Bild...</div>
                    <div class="model-loading-progress">
                        <div class="progress-bar-container">
                            <div class="progress-bar" id="image-progress-bar-${loadingId}"></div>
                        </div>
                        <div class="progress-text" id="image-progress-text-${loadingId}">0%</div>
                    </div>
                `;
                progressBar = document.getElementById(`image-progress-bar-${loadingId}`);
                progressText = document.getElementById(`image-progress-text-${loadingId}`);
            }
        }
        
        // Starte Fortschritts-Simulation (da wir keine echten Updates vom Backend bekommen)
        const numSteps = requestBody.num_inference_steps || 20;
        let currentStep = 0;
        if (progressBar && progressText) {
            progressInterval = setInterval(() => {
                currentStep++;
                const progress = Math.min((currentStep / numSteps) * 100, 95); // Max 95% bis fertig
                progressBar.style.width = `${progress}%`;
                progressText.textContent = `${Math.round(progress)}% (${currentStep}/${numSteps} Schritte)`;
            }, 1000); // Update alle Sekunde
        }
        
        while (retries < maxRetries) {
            try {
                
                response = await apiCall('/image/generate', {
                    method: 'POST',
                    body: JSON.stringify(requestBody),
                    signal: currentImageAbortController.signal
                });
                break; // Erfolgreich - verlasse Schleife
            } catch (error) {
                // Prüfe ob es ein ModelLoading-Fehler ist
                if (error.name === "ModelLoading" || (error.detail && error.detail.status === "model_loading")) {
                    const modelId = error.detail?.model_id || error.detail?.detail?.model_id;
                    if (modelId) {
                        // Zeige Lade-Status
                        loadingMsg = document.getElementById(loadingId);
                        if (loadingMsg) {
                            loadingMsg.querySelector('.message-content').textContent = `Lade Bildmodell ${modelId}...`;
                        }
                        
                        // Warte auf Modell-Laden
                        await waitForModelLoad("image", modelId, currentConversationId);
                        
                        // Wiederhole Request
                        retries++;
                        continue;
                    }
                }
                // Anderer Fehler - weiterwerfen
                throw error;
            }
        }
        
        if (!response) {
            throw new Error("Maximale Anzahl von Wiederholungen erreicht");
        }
        
        // Stoppe Fortschritts-Simulation
        if (progressInterval) {
            clearInterval(progressInterval);
        }
        
        // Remove loading message
        loadingMsg = document.getElementById(loadingId);
        if (loadingMsg) {
            loadingMsg.remove();
        }
        
        // Display image
        const imageUrl = `data:image/png;base64,${response.image_base64}`;
        addImageToChat(prompt, imageUrl);
        
        // Zeige GPU-Status-Informationen falls vorhanden
        if (response.auto_resized || response.cpu_offload_used) {
            let statusMsg = 'Bild generiert';
            if (response.auto_resized) {
                statusMsg += ` (Größe automatisch angepasst: ${response.width}x${response.height})`;
            }
            if (response.cpu_offload_used) {
                statusMsg += ' [CPU-Offload aktiviert]';
            }
            updateModelStatus('image', statusMsg);
            
            // Zeige Info-Meldung
            const infoMsg = document.createElement('div');
            infoMsg.className = 'message assistant';
            infoMsg.style.marginTop = '10px';
            infoMsg.style.padding = '10px';
            infoMsg.style.background = 'var(--bg-tertiary)';
            infoMsg.style.borderRadius = '8px';
            infoMsg.style.fontSize = '0.9em';
            let infoText = '';
            if (response.auto_resized) {
                infoText += `⚠️ Bildgröße wurde automatisch auf ${response.width}x${response.height} reduziert (GPU-Speicher). `;
            }
            if (response.cpu_offload_used) {
                infoText += 'ℹ️ CPU-Offload wurde verwendet, um GPU-Speicher zu sparen.';
            }
            infoMsg.textContent = infoText;
            const chatMessages = document.getElementById('chatMessages');
            if (chatMessages) {
                chatMessages.appendChild(infoMsg);
            }
        } else {
            updateModelStatus('image', 'Bild generiert');
        }
        
        // Clear input
        imagePromptInput.value = '';
        btnGenerateImage.disabled = true;
        
        // Reload conversation to show saved image
        if (currentConversationId) {
            loadConversation(currentConversationId);
        }
        
    } catch (error) {
        // Stoppe Fortschritts-Simulation
        if (progressInterval) {
            clearInterval(progressInterval);
        }
        
        if (error.name === 'AbortError') {
            loadingMsg = document.getElementById(loadingId);
            if (loadingMsg) {
                loadingMsg.remove();
            }
            addMessageToChat('assistant', 'Bildgenerierung abgebrochen.');
            updateModelStatus('image', 'Abgebrochen');
        } else {
            loadingMsg = document.getElementById(loadingId);
            if (loadingMsg) {
                loadingMsg.remove();
            }
            addMessageToChat('assistant', 'Fehler: ' + error.message);
            console.error('Fehler bei Bildgenerierung:', error);
            updateModelStatus('image', 'Fehler');
        }
    } finally {
        // Reset UI
        imagePromptInput.disabled = false;
        btnGenerateImage.style.display = 'block';
        btnCancelImage.style.display = 'none';
        btnGenerateImage.disabled = imagePromptInput.value.trim() === '';
        currentImageAbortController = null;
    }
}

function cancelImageGeneration() {
    if (currentImageAbortController) {
        currentImageAbortController.abort();
        updateModelStatus('image', 'Abgebrochen');
    }
}

function addImageToChat(prompt, imageUrl) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    const timestamp = new Date().toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' });
    
    messageDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="image-prompt">${escapeHtml(prompt)}</div>
            <img src="${imageUrl}" alt="Generated image" class="generated-image" />
            <div class="message-timestamp">${timestamp}</div>
        </div>
    `;
    
    // Füge Click-Event für Lightbox hinzu
    const img = messageDiv.querySelector('.generated-image');
    if (img) {
        img.addEventListener('click', () => {
            showImageLightbox(imageUrl, prompt);
        });
    }
    
    // Remove welcome message if exists
    const welcomeMsg = chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showImageLightbox(imageUrl, prompt) {
    // Erstelle Lightbox-Overlay
    const lightbox = document.createElement('div');
    lightbox.className = 'image-lightbox';
    lightbox.innerHTML = `
        <div class="lightbox-overlay"></div>
        <div class="lightbox-content">
            <div class="lightbox-header">
                <div class="lightbox-prompt">${escapeHtml(prompt)}</div>
                <button class="lightbox-close">×</button>
            </div>
            <img src="${imageUrl}" alt="Generated image" class="lightbox-image" />
            <div class="lightbox-actions">
                <button class="btn-download" onclick="downloadImage('${imageUrl}', '${escapeHtml(prompt)}')">💾 Download</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(lightbox);
    
    // Close on overlay click
    const overlay = lightbox.querySelector('.lightbox-overlay');
    const closeBtn = lightbox.querySelector('.lightbox-close');
    
    const closeLightbox = () => {
        lightbox.remove();
    };
    
    overlay.addEventListener('click', closeLightbox);
    closeBtn.addEventListener('click', closeLightbox);
    
    // Close on ESC key
    const handleEsc = (e) => {
        if (e.key === 'Escape') {
            closeLightbox();
            document.removeEventListener('keydown', handleEsc);
        }
    };
    document.addEventListener('keydown', handleEsc);
}

function downloadImage(imageUrl, prompt) {
    // Erstelle Download-Link
    const link = document.createElement('a');
    link.href = imageUrl;
    const filename = `${prompt.substring(0, 30).replace(/[^a-z0-9]/gi, '_')}_${Date.now()}.png`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function addMessageToChat(role, content, isLoading = false, files = []) {
    const messageId = 'msg-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.id = isLoading ? messageId : undefined;
    
    const avatar = role === 'user' ? 'U' : 'AI';
    const timestamp = new Date().toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' });
    
    let filePreviews = '';
    if (files && files.length > 0) {
        filePreviews = files.map(file => {
            if (file.type === 'csv' && file.preview) {
                return `<div class="file-preview">
                    <div class="file-preview-header">
                        <span>${escapeHtml(file.name)} (${file.size} Zeilen)</span>
                    </div>
                    <div class="file-preview-content">${file.preview}</div>
                </div>`;
            } else {
                return `<div class="file-preview">
                    <div class="file-preview-header">
                        <span>${escapeHtml(file.name)}</span>
                    </div>
                    <div class="file-preview-content">${escapeHtml(file.preview || file.content.substring(0, 500))}${file.content.length > 500 ? '...' : ''}</div>
                </div>`;
            }
        }).join('');
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            ${formatMessageContent(content)}
            ${filePreviews}
            ${!isLoading ? `<div class="message-timestamp">${timestamp}</div>` : ''}
        </div>
    `;
    
    // Remove welcome message if exists
    const welcomeMsg = chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Initialisiere Copy-Buttons für Code-Blöcke
    initializeCodeCopyButtons(messageDiv);
    
    return messageId;
}

async function applyPerformanceSettings() {
    try {
        const cpuThreads = parseInt(cpuThreadsSlider.value);
        const gpuOptimization = gpuOptimizationSelect.value;
        const disableOffload = disableCpuOffload.checked;
        const primaryBudget = primaryBudgetSlider ? parseFloat(primaryBudgetSlider.value) : null;
        
        const settingsData = {
            cpu_threads: cpuThreads === 0 ? null : cpuThreads,
            gpu_optimization: gpuOptimization,
            disable_cpu_offload: disableOffload
        };
        
        // Add GPU allocation settings if slider exists
        if (primaryBudgetSlider) {
            const maxPercent = parseFloat(gpuMaxPercent?.textContent || '90');
            if (primaryBudget > 0 && primaryBudget <= maxPercent) {
                settingsData.primary_budget_percent = primaryBudget;
            } else if (primaryBudget === 0) {
                settingsData.primary_budget_percent = null; // Auto
            } else {
                throw new Error(`Primary Budget muss zwischen 0 und ${maxPercent}% sein`);
            }
        }
        
        await apiCall('/performance/settings', {
            method: 'POST',
            body: JSON.stringify(settingsData)
        });
        
        alert('Performance-Einstellungen wurden gespeichert. Sie werden beim nächsten Modell-Laden wirksam.');
    } catch (error) {
        console.error('Fehler beim Anwenden der Performance-Einstellungen:', error);
        alert('Fehler: ' + error.message);
    }
}

async function loadPerformanceSettings() {
    try {
        const data = await apiCall('/performance/settings');
        if (data.cpu_threads !== undefined) {
            cpuThreadsSlider.value = data.cpu_threads || 0;
            cpuThreadsValue.textContent = data.cpu_threads === 0 || !data.cpu_threads ? 'Auto' : data.cpu_threads;
        }
        if (data.gpu_optimization) {
            gpuOptimizationSelect.value = data.gpu_optimization;
        }
        if (data.disable_cpu_offload !== undefined) {
            disableCpuOffload.checked = data.disable_cpu_offload;
        }
        // Load GPU allocation settings
        if (data.gpu_max_percent !== undefined && gpuMaxPercent) {
            gpuMaxPercent.textContent = data.gpu_max_percent.toFixed(0);
            if (primaryBudgetSlider) {
                primaryBudgetSlider.max = data.gpu_max_percent;
            }
        }
        if (data.primary_budget_percent !== undefined && primaryBudgetSlider && primaryBudgetValue) {
            if (data.primary_budget_percent === null || data.primary_budget_percent === 0) {
                primaryBudgetSlider.value = 0;
                primaryBudgetValue.textContent = 'Auto';
            } else {
                primaryBudgetSlider.value = data.primary_budget_percent;
                primaryBudgetValue.textContent = `${data.primary_budget_percent}%`;
            }
        }
        // max_length für Chat (UI/Streaming Progress)
        if (data.max_length !== undefined && data.max_length !== null) {
            settings.max_length = data.max_length;
        }
        // temperature ggf. ebenfalls aus Performance Settings übernehmen, wenn UI noch nicht verändert wurde
        if (data.temperature !== undefined && data.temperature !== null) {
            // nicht hart überschreiben, wenn User bereits Slider bewegt hat
            if (typeof settings.temperature !== 'number' || isNaN(settings.temperature)) {
                settings.temperature = data.temperature;
            }
        }
    } catch (error) {
        console.error('Fehler beim Laden der Performance-Einstellungen:', error);
    }
}

async function loadPreferences() {
    try {
        const prefs = await apiCall('/preferences');
        preferenceToggle.checked = prefs.enabled;
        settings.preferenceLearning = prefs.enabled;
    } catch (error) {
        console.error('Fehler beim Laden der Präferenzen:', error);
    }
}

async function togglePreferenceLearning() {
    try {
        const response = await apiCall('/preferences/toggle', { method: 'POST' });
        settings.preferenceLearning = response.enabled;
    } catch (error) {
        console.error('Fehler beim Toggle:', error);
        preferenceToggle.checked = !preferenceToggle.checked; // Revert
    }
}

async function resetPreferences() {
    if (!confirm('Präferenzen wirklich zurücksetzen?')) return;
    
    try {
        await apiCall('/preferences/reset', { method: 'POST' });
        alert('Präferenzen zurückgesetzt');
    } catch (error) {
        console.error('Fehler beim Zurücksetzen:', error);
        alert('Fehler beim Zurücksetzen: ' + error.message);
    }
}

// Audio-Einstellungen
async function loadAudioSettings() {
    try {
        const data = await apiCall('/audio/settings');
        if (data.transcription_language !== undefined) {
            settings.transcriptionLanguage = data.transcription_language || "";
            const transcriptionLanguageSelect = document.getElementById('transcriptionLanguageSelect');
            if (transcriptionLanguageSelect) {
                transcriptionLanguageSelect.value = settings.transcriptionLanguage;
            }
        }
    } catch (error) {
        console.error('Fehler beim Laden der Audio-Einstellungen:', error);
    }
}

async function saveAudioSettings() {
    try {
        await apiCall('/audio/settings', {
            method: 'POST',
            body: JSON.stringify({
                transcription_language: settings.transcriptionLanguage || ""
            })
        });
    } catch (error) {
        console.error('Fehler beim Speichern der Audio-Einstellungen:', error);
    }
}

// Output-Einstellungen
async function loadOutputSettings() {
    try {
        const data = await apiCall('/output/settings');
        const outputBaseDirectory = document.getElementById('outputBaseDirectory');
        const outputUseDateFolders = document.getElementById('outputUseDateFolders');
        const outputFilenameFormat = document.getElementById('outputFilenameFormat');
        
        if (outputBaseDirectory && data.base_directory) {
            outputBaseDirectory.value = data.base_directory;
        }
        if (outputUseDateFolders && data.use_date_folders !== undefined) {
            outputUseDateFolders.checked = data.use_date_folders;
        }
        if (outputFilenameFormat && data.filename_format) {
            outputFilenameFormat.value = data.filename_format;
        }
    } catch (error) {
        console.error('Fehler beim Laden der Output-Einstellungen:', error);
    }
}

async function saveOutputSettings() {
    try {
        const outputBaseDirectory = document.getElementById('outputBaseDirectory');
        const outputUseDateFolders = document.getElementById('outputUseDateFolders');
        const outputFilenameFormat = document.getElementById('outputFilenameFormat');
        
        const response = await apiCall('/output/settings', {
            method: 'POST',
            body: JSON.stringify({
                base_directory: outputBaseDirectory ? outputBaseDirectory.value : null,
                use_date_folders: outputUseDateFolders ? outputUseDateFolders.checked : null,
                filename_format: outputFilenameFormat ? outputFilenameFormat.value : null
            })
        });
        
        if (response) {
            alert('Output-Einstellungen erfolgreich gespeichert!');
        }
    } catch (error) {
        console.error('Fehler beim Speichern der Output-Einstellungen:', error);
        alert('Fehler beim Speichern: ' + error.message);
    }
}

function openOutputFolder() {
    const outputBaseDirectory = document.getElementById('outputBaseDirectory');
    if (outputBaseDirectory && outputBaseDirectory.value) {
        // Versuche den Ordner zu öffnen
        // Im Browser können wir nicht direkt OS-Ordner öffnen,
        // aber wir können eine Benachrichtigung anzeigen
        alert(`Output-Ordner:\n${outputBaseDirectory.value}\n\nBitte öffnen Sie diesen Ordner im Windows Explorer.`);
        
        // Kopiere Pfad in Zwischenablage falls möglich
        if (navigator.clipboard) {
            navigator.clipboard.writeText(outputBaseDirectory.value).then(() => {
                console.log('Pfad in Zwischenablage kopiert');
            }).catch(err => {
                console.error('Fehler beim Kopieren:', err);
            });
        }
    }
}

// Agent-Modus Funktionen
async function toggleAgentMode() {
    if (!currentConversationId) {
        alert('Bitte wählen Sie zuerst eine Conversation aus');
        return;
    }
    
    try {
        // Hole aktuellen Status
        const currentStatus = await loadAgentMode();
        const newStatus = !currentStatus;
        
        // Setze neuen Status
        const response = await apiCall(`/conversations/${currentConversationId}/agent-mode`, {
            method: 'POST',
            body: JSON.stringify({ enabled: newStatus })
        });
        
        // Update Button
        updateAgentModeButton(newStatus);
    } catch (error) {
        console.error('Fehler beim Toggle Agent-Modus:', error);
        alert('Fehler beim Ändern des Agent-Modus: ' + error.message);
    }
}

async function loadAgentMode() {
    if (!currentConversationId) {
        return false;
    }
    
    try {
        const response = await apiCall(`/conversations/${currentConversationId}/agent-mode`);
        const agentMode = response.agent_mode || false;
        updateAgentModeButton(agentMode);
        return agentMode;
    } catch (error) {
        console.error('Fehler beim Laden des Agent-Modus:', error);
        updateAgentModeButton(false);
        return false;
    }
}

function updateAgentModeButton(enabled) {
    if (!btnAgentMode) return;
    
    if (enabled) {
        btnAgentMode.classList.add('active');
        btnAgentMode.title = 'Agent-Modus aktiviert (WebSearch & Dateimanipulation)';
    } else {
        btnAgentMode.classList.remove('active');
        btnAgentMode.title = 'Agent-Modus deaktiviert - Klicken zum Aktivieren';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function initializeCodeCopyButtons(container) {
    /**
     * Initialisiert Copy-Buttons für alle Code-Blöcke in einem Container
     */
    const copyButtons = container.querySelectorAll('.code-copy-btn');
    
    copyButtons.forEach(button => {
        // Entferne alte Event-Listener (falls vorhanden)
        const newButton = button.cloneNode(true);
        button.parentNode.replaceChild(newButton, button);
        
        // Füge neuen Event-Listener hinzu
        newButton.addEventListener('click', async () => {
            const codeId = newButton.getAttribute('data-code-id');
            const codeElement = document.getElementById(codeId);
            
            if (!codeElement) return;
            
            const codeText = codeElement.textContent;
            
            try {
                await navigator.clipboard.writeText(codeText);
                
                // Visuelles Feedback
                const copyText = newButton.querySelector('.copy-text');
                const originalText = copyText.textContent;
                copyText.textContent = 'Kopiert!';
                newButton.classList.add('copied');
                
                setTimeout(() => {
                    copyText.textContent = originalText;
                    newButton.classList.remove('copied');
                }, 2000);
            } catch (err) {
                console.error('Fehler beim Kopieren:', err);
                // Fallback: Alte Methode
                const textArea = document.createElement('textarea');
                textArea.value = codeText;
                textArea.style.position = 'fixed';
                textArea.style.opacity = '0';
                document.body.appendChild(textArea);
                textArea.select();
                try {
                    document.execCommand('copy');
                    const copyText = newButton.querySelector('.copy-text');
                    copyText.textContent = 'Kopiert!';
                    setTimeout(() => {
                        copyText.textContent = 'Kopieren';
                    }, 2000);
                } catch (e) {
                    console.error('Fallback-Kopieren fehlgeschlagen:', e);
                }
                document.body.removeChild(textArea);
            }
        });
    });
}

function formatMessageContent(text) {
    /**
     * Formatiert Nachrichtentext mit klickbaren Links, Markdown-Unterstützung und Code-Blöcken
     * - Escaped HTML für XSS-Schutz
     * - Macht URLs klickbar
     * - Unterstützt Markdown-Links [text](url)
     * - Erkennt Code-Blöcke (```language ... ```) und macht sie kopierbar
     * - Erhält Zeilenumbrüche
     * - Unterstützt HTML-Quellen-Header (wird nicht escaped)
     */
    
    const normalizedText = unwrapMarkdownFence(text);
    
    // FIX: HTML-Quellen-Header extrahieren (bevor HTML escaped wird)
    const sourcesHeaders = [];
    let sourcesHeaderIndex = 0;
    // Pattern: <div style='...'><strong>Quellen:</strong> ... </div>
    const sourcesHeaderPattern = /<div[^>]*style=['"][^'"]*['"][^>]*><strong>Quellen:<\/strong>[^<]*<\/div>\s*\n*/g;
    let processedText = normalizedText.replace(sourcesHeaderPattern, (match) => {
        const headerId = `sources-header-${sourcesHeaderIndex++}`;
        sourcesHeaders.push({
            id: headerId,
            html: match.trim()
        });
        return `__SOURCES_HEADER_${headerId}__`;
    });
    
    // Code-Blöcke zuerst extrahieren (bevor HTML escaped wird)
    const codeBlocks = [];
    let codeBlockIndex = 0;
    
    // Pattern: ```language\ncode\n``` oder ```language code\n``` oder ```\ncode\n```
    // Unterstützt auch Code-Blöcke ohne Newline nach der Sprache
    const codeBlockPattern = /```(\w+)?\s*\n?([\s\S]*?)```/g;
    processedText = processedText.replace(codeBlockPattern, (match, language, code) => {
        const blockId = `code-block-${codeBlockIndex++}`;
        // Entferne führende/trailing Whitespace und Newlines
        const cleanCode = code.trim();
        codeBlocks.push({
            id: blockId,
            language: (language || 'text').trim(),
            code: cleanCode
        });
        return `__CODE_BLOCK_${blockId}__`;
    });
    
    // Escape HTML (XSS-Schutz) - aber Code-Blöcke und Quellen-Header sind bereits extrahiert
    let escaped = escapeHtml(processedText);
    
    // Markdown-Links erkennen und umwandeln: [text](url) (VOR Code-Blöcken einfügen)
    // Pattern: [beliebiger Text](http://... oder https://...)
    escaped = escaped.replace(
        /\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer" class="message-link">$1</a>'
    );
    
    // Normale URLs erkennen und klickbar machen (die nicht bereits in einem Link sind)
    // Pattern: http:// oder https:// gefolgt von nicht-whitespace Zeichen
    escaped = escaped.replace(
        /(?<!href=["'])(?<!">)(https?:\/\/[^\s<>"]+)/g,
        '<a href="$1" target="_blank" rel="noopener noreferrer" class="message-link">$1</a>'
    );
    
    // Markdown-Basis-Rendering (Überschriften, Listen, Fett/Kursiv)
    escaped = renderMarkdownBasic(escaped);
    
    // FIX: Quellen-Header wieder einfügen (VOR Code-Blöcken, damit sie oben stehen)
    sourcesHeaders.forEach(header => {
        escaped = escaped.replace(`__SOURCES_HEADER_${header.id}__`, header.html);
    });
    
    // Code-Blöcke wieder einfügen mit Copy-Button (NACH Markdown-Rendering)
    codeBlocks.forEach(block => {
        const codeId = `code-content-${block.id}`;
        const copyButtonId = `copy-btn-${block.id}`;
        // Code mit Zeilenumbrüchen beibehalten (für <pre>)
        const codeWithBreaks = escapeHtml(block.code).replace(/\n/g, '\n');
        const codeHtml = `
            <div class="code-block-container">
                <div class="code-block-header">
                    <span class="code-language">${escapeHtml(block.language)}</span>
                    <button class="code-copy-btn" id="${copyButtonId}" data-code-id="${codeId}" title="Code kopieren">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        <span class="copy-text">Kopieren</span>
                    </button>
                </div>
                <pre class="code-block"><code id="${codeId}" class="language-${escapeHtml(block.language)}">${codeWithBreaks}</code></pre>
            </div>
        `;
        escaped = escaped.replace(`__CODE_BLOCK_${block.id}__`, codeHtml);
    });
    
    return escaped;
}

function unwrapMarkdownFence(text) {
    /**
     * Entfernt ein einzelnes äußeres Markdown-Fence, wenn es nur als
     * Container für Markdown-Beispiele genutzt wird.
     */
    const trimmed = text.trim();
    const match = trimmed.match(/^```(\w+)?\s*\n([\s\S]*?)\n```$/);
    if (!match) {
        return text;
    }
    
    const language = (match[1] || '').toLowerCase();
    if (language && !['markdown', 'md'].includes(language)) {
        return text;
    }
    
    const inner = match[2];
    const markdownSignals = [
        /^#{1,6}\s+/m,
        /^\s*[-*]\s+/m,
        /^\s*\d+\.\s+/m,
        /^\s*>\s+/m,
        /\[[^\]]+\]\([^)]+\)/m,
        /!\[[^\]]*\]\([^)]+\)/m,
        /^\s*\|.+\|\s*$/m
    ];
    
    if (!markdownSignals.some((pattern) => pattern.test(inner))) {
        return text;
    }
    
    return inner.trim();
}

function renderMarkdownBasic(text) {
    /**
     * Minimaler Markdown-Renderer (sicher, ohne HTML-Interpretation).
     * Unterstützt:
     * - Überschriften (#..######)
     * - Ungeordnete/Geordnete Listen
     * - Fett/Kursiv inline
     */
    const lines = text.split('\n');
    const output = [];
    let inUl = false;
    let inOl = false;
    
    const closeLists = () => {
        if (inUl) {
            output.push('</ul>');
            inUl = false;
        }
        if (inOl) {
            output.push('</ol>');
            inOl = false;
        }
    };
    
    const renderInline = (line) => {
        return line
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/\*([^*]+)\*/g, '<em>$1</em>');
    };
    
    for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) {
            closeLists();
            output.push('<br>');
            continue;
        }
        
        if (line.startsWith('__CODE_BLOCK_') || line.startsWith('__SOURCES_HEADER_')) {
            closeLists();
            output.push(line);
            continue;
        }
        
        const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
        if (headingMatch) {
            closeLists();
            const level = headingMatch[1].length;
            const content = renderInline(headingMatch[2]);
            output.push(`<h${level}>${content}</h${level}>`);
            continue;
        }
        
        const ulMatch = line.match(/^[\-\*]\s+(.*)$/);
        if (ulMatch) {
            if (inOl) {
                output.push('</ol>');
                inOl = false;
            }
            if (!inUl) {
                output.push('<ul>');
                inUl = true;
            }
            output.push(`<li>${renderInline(ulMatch[1])}</li>`);
            continue;
        }
        
        const olMatch = line.match(/^\d+\.\s+(.*)$/);
        if (olMatch) {
            if (inUl) {
                output.push('</ul>');
                inUl = false;
            }
            if (!inOl) {
                output.push('<ol>');
                inOl = true;
            }
            output.push(`<li>${renderInline(olMatch[1])}</li>`);
            continue;
        }
        
        closeLists();
            output.push(`<p>${renderInline(line)}</p>`);
    }
    
    closeLists();
    return output.join('');
}

// System Stats Update
let statsUpdateInterval = null;
let lastStats = {
    cpu: null,
    ram: null,
    gpu: null
};
let statsBackoffDelay = 2000; // Start mit 2 Sekunden
let statsConsecutiveErrors = 0;
let statsPaused = false;
const MAX_BACKOFF_DELAY = 30000; // Maximal 30 Sekunden
const MAX_CONSECUTIVE_ERRORS = 5; // Nach 5 Fehlern pausieren

function startSystemStatsUpdate() {
    // Prüfe ob Elemente existieren
    if (!cpuPercent || !ramPercent || !gpuPercent) {
        console.error('System-Stats Elemente nicht gefunden:', {
            cpuPercent: !!cpuPercent,
            ramPercent: !!ramPercent,
            gpuPercent: !!gpuPercent
        });
        return;
    }
    
    console.log('Starte System-Stats Update');
    updateSystemStats();
    // Starte Polling mit variablem Intervall
    scheduleNextStatsUpdate();
}

function scheduleNextStatsUpdate() {
    if (statsUpdateInterval) {
        clearInterval(statsUpdateInterval);
    }
    
    if (statsPaused) {
        // Versuche alle 10 Sekunden wieder zu verbinden
        statsUpdateInterval = setTimeout(() => {
            statsPaused = false;
            statsConsecutiveErrors = 0;
            statsBackoffDelay = 2000;
            updateSystemStats();
            scheduleNextStatsUpdate();
        }, 10000);
        return;
    }
    
    statsUpdateInterval = setTimeout(() => {
        updateSystemStats();
        scheduleNextStatsUpdate();
    }, statsBackoffDelay);
}

async function updateSystemStats() {
    try {
        const stats = await apiCall('/system/stats', { method: 'GET' });
        
        if (!stats) {
            console.warn('Keine Stats-Daten erhalten');
            statsConsecutiveErrors++;
            handleStatsError();
            return;
        }
        
        // Erfolgreich - Reset Backoff
        statsConsecutiveErrors = 0;
        statsBackoffDelay = 2000; // Zurück zu normalem Intervall
        statsPaused = false;
        
        // CPU
        if (stats.cpu_percent !== null && stats.cpu_percent !== undefined) {
            const cpuValue = Math.round(stats.cpu_percent);
            if (cpuPercent) {
                cpuPercent.textContent = `${cpuValue}%`;
                lastStats.cpu = cpuValue;
            }
        }
        
        // RAM
        if (stats.ram_percent !== null && stats.ram_percent !== undefined) {
            const ramValue = Math.round(stats.ram_percent);
            if (ramPercent) {
                ramPercent.textContent = `${ramValue}%`;
                lastStats.ram = ramValue;
            }
        }
        
        // GPU
        if (gpuPercent) {
            if (stats.gpu_available) {
                if (stats.gpu_utilization !== null && stats.gpu_utilization !== undefined) {
                    const gpuValue = stats.gpu_utilization;
                    gpuPercent.textContent = `${gpuValue}%`;
                    lastStats.gpu = gpuValue;
                } else if (stats.gpu_memory_percent !== null && stats.gpu_memory_percent !== undefined) {
                    const gpuValue = Math.round(stats.gpu_memory_percent);
                    gpuPercent.textContent = `${gpuValue}%`;
                    lastStats.gpu = gpuValue;
                } else {
                    gpuPercent.textContent = 'OK';
                    lastStats.gpu = 'OK';
                }
            } else {
                gpuPercent.textContent = 'CPU';
                lastStats.gpu = 'CPU';
            }
        }
    } catch (error) {
        statsConsecutiveErrors++;
        handleStatsError(error);
    }
}

function handleStatsError(error = null) {
    const isNetworkError = error && (
        error.message && (
            error.message.includes('fetch') || 
            error.message.includes('Netzwerkfehler') || 
            error.message.includes('CONNECTION_REFUSED')
        )
    );
    
    // Exponential Backoff
    if (statsConsecutiveErrors > 0) {
        statsBackoffDelay = Math.min(
            statsBackoffDelay * 1.5,
            MAX_BACKOFF_DELAY
        );
    }
    
    // Pausiere nach zu vielen Fehlern
    if (statsConsecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        statsPaused = true;
        // Zeige visuelles Feedback (optional - kann entfernt werden wenn störend)
        if (cpuPercent && lastStats.cpu !== null) {
            cpuPercent.textContent = lastStats.cpu + '%';
        }
    }
    
    // Werte NICHT zurücksetzen - letzte Werte beibehalten
    // Nur wenn noch keine Werte vorhanden sind, "-" anzeigen
    if (cpuPercent && lastStats.cpu === null) cpuPercent.textContent = '-';
    if (ramPercent && lastStats.ram === null) ramPercent.textContent = '-';
    if (gpuPercent && lastStats.gpu === null) gpuPercent.textContent = '-';
    
    // Logge nur nicht-Netzwerkfehler
    if (error && !isNetworkError) {
        console.error('Fehler beim Abrufen der System-Stats:', error);
    }
}

// File Upload Functions
function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    handleFiles(files);
    fileInput.value = ''; // Reset input
}

async function handleFiles(files) {
    for (const file of files) {
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            alert(`Datei ${file.name} ist zu groß (max. 10MB)`);
            continue;
        }
        
        const fileData = {
            name: file.name,
            type: file.name.split('.').pop().toLowerCase(),
            size: file.size,
            content: '',
            preview: ''
        };
        
        try {
            const text = await file.text();
            fileData.content = text;
            
            // Parse CSV
            if (fileData.type === 'csv') {
                fileData.preview = parseCSVPreview(text);
            } else {
                // Preview für andere Dateitypen
                const lines = text.split('\n');
                fileData.preview = lines.slice(0, 20).join('\n');
                if (lines.length > 20) {
                    fileData.preview += `\n... (${lines.length - 20} weitere Zeilen)`;
                }
            }
            
            uploadedFilesList.push(fileData);
        } catch (error) {
            console.error('Fehler beim Lesen der Datei:', error);
            alert(`Fehler beim Lesen der Datei ${file.name}: ${error.message}`);
        }
    }
    
    renderUploadedFiles();
}

function parseCSVPreview(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim());
    if (lines.length === 0) return 'Leere CSV-Datei';
    
    const maxRows = 10;
    const previewLines = lines.slice(0, maxRows);
    const hasMore = lines.length > maxRows;
    
    // Verbessertes CSV-Parsing
    function parseCSVLine(line) {
        const values = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            
            if (char === '"') {
                if (inQuotes && line[i + 1] === '"') {
                    // Escaped quote
                    current += '"';
                    i++; // Skip next quote
                } else {
                    // Toggle quote state
                    inQuotes = !inQuotes;
                }
            } else if (char === ',' && !inQuotes) {
                // End of field
                values.push(current.trim());
                current = '';
            } else {
                current += char;
            }
        }
        // Add last field
        values.push(current.trim());
        
        return values;
    }
    
    const rows = previewLines.map(line => parseCSVLine(line));
    
    if (rows.length === 0) return 'Keine Daten';
    
    // Finde maximale Spaltenanzahl
    const maxCols = Math.max(...rows.map(r => r.length));
    
    // Erstelle HTML-Tabelle
    let html = '<table class="file-preview-table">';
    
    // Header (erste Zeile)
    if (rows.length > 0) {
        html += '<thead><tr>';
        for (let i = 0; i < maxCols; i++) {
            const header = rows[0][i] || `Spalte ${i + 1}`;
            html += `<th>${escapeHtml(header)}</th>`;
        }
        html += '</tr></thead>';
    }
    
    // Body (restliche Zeilen)
    html += '<tbody>';
    rows.slice(1).forEach(row => {
        html += '<tr>';
        for (let i = 0; i < maxCols; i++) {
            const cell = row[i] || '';
            html += `<td>${escapeHtml(cell)}</td>`;
        }
        html += '</tr>';
    });
    html += '</tbody></table>';
    
    if (hasMore) {
        html += `<div style="margin-top: 8px; font-size: 11px; color: var(--text-secondary);">... (${lines.length - maxRows} weitere Zeilen)</div>`;
    }
    
    return html;
}

function renderUploadedFiles() {
    uploadedFiles.innerHTML = '';
    
    uploadedFilesList.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'uploaded-file-item';
        fileItem.innerHTML = `
            <span class="file-name">${escapeHtml(file.name)}</span>
            <button class="file-remove" onclick="removeFile(${index})" title="Entfernen">×</button>
        `;
        uploadedFiles.appendChild(fileItem);
    });
    
    // Enable send button if files are uploaded
    if (uploadedFilesList.length > 0) {
        btnSend.disabled = false;
    }
}

function removeFile(index) {
    uploadedFilesList.splice(index, 1);
    renderUploadedFiles();
    if (uploadedFilesList.length === 0 && messageInput.value.trim() === '') {
        btnSend.disabled = true;
    }
}

// Kein automatisches Polling mehr - Status wird nur bei Bedarf aktualisiert

// Audio Recording
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

async function toggleMicrophone() {
    console.log('toggleMicrophone aufgerufen, isRecording:', isRecording);
    if (!isRecording) {
        await startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        // Prüfe ob Browser Mikrofon-Zugriff unterstützt
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Ihr Browser unterstützt keine Mikrofon-Aufnahme. Bitte verwenden Sie einen modernen Browser wie Chrome, Firefox oder Edge.');
            return;
        }
        
        // Frage nach Mikrofon-Berechtigung
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Erstelle MediaRecorder
        mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });
        
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            // Konvertiere zu WAV und sende an Server
            await processAudioRecording();
            
            // Stream stoppen
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        isRecording = true;
        btnMicrophone.classList.add('recording');
        btnMicrophone.textContent = '⏹️';
        statusText.textContent = 'Aufnahme läuft...';
        
    } catch (error) {
        console.error('Fehler beim Starten der Aufnahme:', error);
        alert('Fehler beim Starten der Aufnahme: ' + error.message);
        isRecording = false;
        btnMicrophone.classList.remove('recording');
        btnMicrophone.textContent = '🎤';
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        btnMicrophone.classList.remove('recording');
        btnMicrophone.textContent = '🎤';
        updateModelStatus('audio', 'Verarbeite Aufnahme...');
    }
}

async function processAudioRecording() {
    try {
        // Erstelle Blob aus Audio-Chunks
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
        
        // Konvertiere WebM zu WAV
        const wavBlob = await convertWebmToWav(audioBlob);
        
        // Sende an Server
        const formData = new FormData();
        formData.append('file', wavBlob, 'recording.wav');
        
        // Hole gespeicherte Sprache aus Einstellungen
        const language = settings.transcriptionLanguage || "";
        
        // Baue URL mit optionalem language-Parameter
        let transcribeUrl = `${API_BASE}/audio/transcribe`;
        if (language) {
            transcribeUrl += `?language=${encodeURIComponent(language)}`;
        }
        
        updateModelStatus('audio', 'Transkribiere...');
        
        let response;
        let retries = 0;
        const maxRetries = 3;
        
        while (retries < maxRetries) {
            try {
                response = await fetch(transcribeUrl, {
                    method: 'POST',
                    body: formData
                });
                
                // Prüfe ob es ein "model_loading" Status ist (202 Accepted)
                if (response.status === 202) {
                    const errorData = await response.json();
                    if (errorData.detail && errorData.detail.status === "model_loading") {
                        const modelId = errorData.detail.model_id;
                        updateModelStatus('audio', `Lade Modell ${modelId}...`);
                        // Warte auf Modell-Laden
                        await waitForModelLoad("audio", modelId);
                        // Wiederhole Request
                        retries++;
                        continue;
                    }
                }
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Transkriptions-Fehler');
                }
                
                // Erfolgreich - verlasse Schleife
                break;
            } catch (error) {
                // Bei ModelLoading-Fehler, wiederhole
                if (error.name === "ModelLoading" || (error.detail && error.detail.status === "model_loading")) {
                    retries++;
                    continue;
                }
                // Anderer Fehler - weiterwerfen
                throw error;
            }
        }
        
        if (!response || !response.ok) {
            throw new Error("Maximale Anzahl von Wiederholungen erreicht oder Transkription fehlgeschlagen");
        }
        
        const data = await response.json();
        
        // Setze transkribierten Text in Input-Feld
        messageInput.value = data.text;
        btnSend.disabled = false;
        autoResizeTextarea();
        
        updateModelStatus('audio', 'Transkription erfolgreich');
        setTimeout(() => {
            updateModelStatus('audio', null);
        }, 2000);
        
    } catch (error) {
        console.error('Fehler bei der Transkription:', error);
        alert('Fehler bei der Transkription: ' + error.message);
        updateModelStatus('audio', 'Fehler');
        setTimeout(() => {
            updateModelStatus('audio', null);
        }, 2000);
    }
}

async function convertWebmToWav(webmBlob) {
    // Verwende Web Audio API für Konvertierung
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    // Lade Audio-Daten
    const arrayBuffer = await webmBlob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    
    // Konvertiere zu 16-bit PCM WAV
    const wavBuffer = audioBufferToWav(audioBuffer);
    return new Blob([wavBuffer], { type: 'audio/wav' });
}

function audioBufferToWav(buffer) {
    const length = buffer.length;
    const numberOfChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const bytesPerSample = 2;
    const blockAlign = numberOfChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = length * blockAlign;
    const bufferSize = 44 + dataSize;
    
    const arrayBuffer = new ArrayBuffer(bufferSize);
    const view = new DataView(arrayBuffer);
    
    // WAV Header schreiben
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, bufferSize - 8, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, 1, true); // audio format (PCM)
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true); // bits per sample
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);
    
    // Audio-Daten schreiben
    let offset = 44;
    for (let i = 0; i < length; i++) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }
    }
    
    return arrayBuffer;
}

// Update-Queue für DOM-Updates (verhindert UI-Blockierung)
class UpdateQueue {
    constructor() {
        this.pending = new Map();
        this.rafId = null;
    }
    
    schedule(key, updateFn) {
        this.pending.set(key, updateFn);
        if (!this.rafId) {
            this.rafId = requestAnimationFrame(() => this.flush());
        }
    }
    
    flush() {
        this.pending.forEach(fn => fn());
        this.pending.clear();
        this.rafId = null;
    }
}

// Scroll-Throttling (verhindert zu häufige Scroll-Updates)
let lastScrollUpdate = 0;
const SCROLL_THROTTLE_MS = 100;

function scheduleScrollUpdate() {
    const now = Date.now();
    if (now - lastScrollUpdate >= SCROLL_THROTTLE_MS) {
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        lastScrollUpdate = now;
    }
}

// Streaming Chat-Funktion
async function sendMessageStream(message, conversationId, loadingId) {
    const streamStartTime = Date.now();
    try {
        const response = await fetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                conversation_id: conversationId,
                temperature: settings.temperature,
                max_length: settings.max_length || 2048
            }),
            signal: currentAbortController?.signal
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        // Entferne Loading-Message
        const loadingMsg = document.getElementById(loadingId);
        if (loadingMsg) {
            loadingMsg.remove();
        }
        
        // Erstelle neue Message für Streaming
        const messageId = 'msg-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        messageDiv.id = messageId;
        
        const avatar = 'AI';
        const timestamp = new Date().toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <span id="streaming-content-${messageId}"></span>
                <div id="tool-status-${messageId}" class="tool-status" style="display: none;"></div>
                <div id="streaming-progress-${messageId}" class="streaming-progress" style="display: none;">
                    <div class="progress-bar-container">
                        <div class="progress-bar" id="progress-bar-${messageId}"></div>
                    </div>
                    <div class="progress-text" id="progress-text-${messageId}">0/0 Tokens (0%)</div>
                </div>
                <div class="message-timestamp">${timestamp}</div>
            </div>
        `;
        
        // Remove welcome message if exists
        const welcomeMsg = chatMessages.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }
        
        chatMessages.appendChild(messageDiv);
        const contentElement = document.getElementById(`streaming-content-${messageId}`);
        const toolStatusElement = document.getElementById(`tool-status-${messageId}`);
        const progressContainer = document.getElementById(`streaming-progress-${messageId}`);
        const progressBar = document.getElementById(`progress-bar-${messageId}`);
        const progressText = document.getElementById(`progress-text-${messageId}`);
        
        let fullResponse = '';
        let tokenCount = 0;
        // max_length für Progress-Berechnung (pro Beitrag)
        const maxTokens = settings.max_length || 2048;
        
        // Erstelle UpdateQueue für DOM-Updates
        const updateQueue = new UpdateQueue();
        let chunkCount = 0;
        
        // Tool status tracking
        const toolStatusMap = {
            'read_file': { icon: '📄', label: 'Lese Datei' },
            'write_file': { icon: '✍️', label: 'Schreibe Datei' },
            'list_directory': { icon: '📁', label: 'Liste Verzeichnis' },
            'delete_file': { icon: '🗑️', label: 'Lösche Datei' },
            'file_exists': { icon: '🔍', label: 'Prüfe Datei' },
            'web_search': { icon: '🌐', label: 'Suche im Web' }
        };
        let activeToolCalls = new Map();
        
        let lastLogTime = Date.now();
        let totalChunks = 0;
        let totalChars = 0;
        
        // Lese Stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const loopStartTime = Date.now();
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        // Handle tool_call event
                        if (data.tool_call) {
                            const tc = data.tool_call;
                            const toolName = tc.name || '';
                            const callId = tc.call_id || '';
                            const toolInfo = toolStatusMap[toolName] || { icon: '🔧', label: toolName };
                            
                            activeToolCalls.set(callId, { name: toolName, startTime: Date.now() });
                            
                            if (toolStatusElement) {
                                const toolLabel = tc.requires_confirmation 
                                    ? `${toolInfo.icon} ${toolInfo.label} (Bestätigung erforderlich)`
                                    : `${toolInfo.icon} ${toolInfo.label}...`;
                                toolStatusElement.innerHTML = `<div class="tool-status-item">${toolLabel}</div>`;
                                toolStatusElement.style.display = 'block';
                            }
                            scheduleScrollUpdate();
                            continue;
                        }
                        
                        // Handle tool_result event
                        if (data.tool_result) {
                            const tr = data.tool_result;
                            const callId = tr.call_id || '';
                            const toolCall = activeToolCalls.get(callId);
                            
                            if (toolCall) {
                                const toolInfo = toolStatusMap[toolCall.name] || { icon: '🔧', label: toolCall.name };
                                const duration = Date.now() - toolCall.startTime;
                                
                                if (tr.ok) {
                                    if (toolStatusElement) {
                                        const existing = toolStatusElement.querySelector(`[data-call-id="${callId}"]`);
                                        if (existing) {
                                            existing.innerHTML = `${toolInfo.icon} ${toolInfo.label} ✓ (${duration}ms)`;
                                            existing.classList.add('tool-success');
                                        } else {
                                            const item = document.createElement('div');
                                            item.className = 'tool-status-item tool-success';
                                            item.setAttribute('data-call-id', callId);
                                            item.innerHTML = `${toolInfo.icon} ${toolInfo.label} ✓ (${duration}ms)`;
                                            toolStatusElement.appendChild(item);
                                        }
                                    }
                                } else {
                                    if (toolStatusElement) {
                                        const existing = toolStatusElement.querySelector(`[data-call-id="${callId}"]`);
                                        if (existing) {
                                            existing.innerHTML = `${toolInfo.icon} ${toolInfo.label} ✗ (${tr.error || 'Fehler'})`;
                                            existing.classList.add('tool-error');
                                        } else {
                                            const item = document.createElement('div');
                                            item.className = 'tool-status-item tool-error';
                                            item.setAttribute('data-call-id', callId);
                                            item.innerHTML = `${toolInfo.icon} ${toolInfo.label} ✗ (${tr.error || 'Fehler'})`;
                                            toolStatusElement.appendChild(item);
                                        }
                                    }
                                }
                                
                                activeToolCalls.delete(callId);
                                
                                // Hide tool status if all tools are done (after a short delay)
                                if (activeToolCalls.size === 0) {
                                    setTimeout(() => {
                                        if (toolStatusElement && activeToolCalls.size === 0) {
                                            toolStatusElement.style.display = 'none';
                                        }
                                    }, 2000);
                                }
                            }
                            scheduleScrollUpdate();
                            continue;
                        }
                        
                        if (data.replace && typeof data.content === 'string') {
                            // Server sendet eine finale, korrigierte Version (z.B. Polish-Rewrite)
                            fullResponse = data.content;
                            if (contentElement) {
                                contentElement.innerHTML = formatMessageContent(fullResponse);
                                initializeCodeCopyButtons(messageDiv);
                            }
                            scheduleScrollUpdate();
                            continue;
                        }
                        if (data.chunk) {
                            const beforeUpdateTime = Date.now();
                            totalChunks++;
                            totalChars += data.chunk.length;
                            
                            fullResponse += data.chunk;
                            // Zähle Tokens grob (ungefähr 1 Token = 4 Zeichen für deutsche Texte)
                            tokenCount = Math.ceil(fullResponse.length / 4);
                            chunkCount++;
                            
                            const contentUpdateStartTime = Date.now();
                            
                            // Content sofort aktualisieren (kritisch für Streaming-Erlebnis)
                            // Verwende innerHTML für Formatierung (Code-Blöcke, Links, etc.)
                            if (contentElement) {
                                contentElement.innerHTML = formatMessageContent(fullResponse);
                                // Initialisiere Copy-Buttons nach jedem Update
                                initializeCodeCopyButtons(messageDiv);
                            }
                            
                            const contentUpdateDuration = Date.now() - contentUpdateStartTime;
                            const timeSinceLastLog = Date.now() - lastLogTime;
                            if (timeSinceLastLog >= 500 || totalChunks === 1) {
                                lastLogTime = Date.now();
                            }
                            
                            // Zeige Fortschrittsanzeige sofort an (nicht gebatchet)
                            if (tokenCount > 0 && progressContainer && progressBar && progressText) {
                                // Container sofort anzeigen (nicht gebatchet)
                                if (progressContainer.style.display === 'none' || !progressContainer.style.display) {
                                    progressContainer.style.display = 'block';
                                }
                                const progressUpdateStartTime = Date.now();
                                
                                // Nur Werte-Updates batchen (für Performance)
                                updateQueue.schedule('progress', () => {
                                    const progress = Math.min((tokenCount / maxTokens) * 100, 100);
                                    progressBar.style.width = `${progress}%`;
                                    progressText.textContent = `${tokenCount}/${maxTokens} Tokens (${Math.round(progress)}%) • ${totalChunks} Chunks`;
                                });
                                
                                const progressUpdateDuration = Date.now() - progressUpdateStartTime;
                                if (progressUpdateDuration > 5) {
                                }
                            }
                            
                            const scrollStartTime = Date.now();
                            
                            // Throttled Scroll-Update
                            scheduleScrollUpdate();
                            
                            const scrollDuration = Date.now() - scrollStartTime;
                            if (scrollDuration > 5) {
                            }
                            
                            const beforeYieldTime = Date.now();
                            
                            // Event Loop Yielding nach jedem 10. Chunk
                            if (chunkCount % 10 === 0) {
                                await new Promise(resolve => setTimeout(resolve, 0));
                            }
                            
                            const yieldDuration = Date.now() - beforeYieldTime;
                            if (yieldDuration > 5 && chunkCount % 10 === 0) {
                            }
                        }
                        if (data.done) {
                            const totalDuration = Date.now() - streamStartTime;
                            
                            // Verstecke Fortschrittsanzeige
                            if (progressContainer) {
                                progressContainer.style.display = 'none';
                            }
                            
                            // Verstecke Tool-Status nach kurzer Verzögerung (falls noch sichtbar)
                            if (toolStatusElement) {
                                setTimeout(() => {
                                    toolStatusElement.style.display = 'none';
                                }, 1500);
                            }
                            
                            // Finale Formatierung mit Code-Blöcken
                            if (contentElement) {
                                contentElement.innerHTML = formatMessageContent(fullResponse);
                                initializeCodeCopyButtons(messageDiv);
                            }
                            
                            // Antwort wird bereits vom Server gespeichert
                            // Update title if first message
                            if (chatTitle.textContent === 'Neues Gespräch') {
                                const firstWords = message.substring(0, 50);
                                updateChatTitle(firstWords + (message.length > 50 ? '...' : ''));
                            }
                            return;
                        }
                        if (data.error) {
                            throw new Error(data.error);
                        }
                    } catch (e) {
                        // Ignoriere Parse-Fehler für unvollständige Chunks
                        if (e.message !== 'Unexpected end of JSON input') {
                            console.warn('Fehler beim Parsen des Stream-Chunks:', e);
                        }
                    }
                }
            }
        }
        
        // Verstecke Fortschrittsanzeige am Ende
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            throw error;
        }
        console.error('Fehler beim Streaming:', error);
        throw error;
    }
}

// Server Restart
async function restartServer() {
    if (!confirm('Möchten Sie den Server wirklich neu starten? Dies wird einige Sekunden dauern.')) {
        return;
    }
    
    try {
        const response = await apiCall('/restart', {
            method: 'POST'
        });
        
        // Zeige Status-Update
        updateModelStatus('text', 'Server startet neu...');
        updateModelStatus('audio', 'Server startet neu...');
        updateModelStatus('image', 'Server startet neu...');
        
        // Warte auf Server-Neustart und aktualisiere Status
        await waitForServerRestart();
        
        // Status aktualisieren
        await loadStatus();
        await loadModelServiceStatus();
        
        alert('Server wurde erfolgreich neu gestartet.');
    } catch (error) {
        console.error('Fehler beim Server-Restart:', error);
        alert('Fehler beim Server-Restart: ' + error.message);
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


