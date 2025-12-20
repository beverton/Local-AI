// API Base URL
const API_BASE = 'http://127.0.0.1:8000';

// State
let currentConversationId = null;
let currentModel = null;
let currentAbortController = null;
let currentImageAbortController = null;
let currentTab = 'chat';
let uploadedFilesList = []; // Liste der hochgeladenen Dateien
let settings = {
    temperature: 0.3,  // Niedriger für bessere Qualität (weniger "Jibberish")
    maxLength: 512,
    preferenceLearning: false
};

// DOM Elements
const conversationsList = document.getElementById('conversationsList');
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const btnSend = document.getElementById('btnSend');
const btnNewChat = document.getElementById('btnNewChat');
const modelSelect = document.getElementById('modelSelect');
const imageModelSelect = document.getElementById('imageModelSelect');
const chatTitle = document.getElementById('chatTitle');
const statusText = document.getElementById('statusText');
const settingsPanel = document.getElementById('settingsPanel');
const btnSettings = document.getElementById('btnSettings');
const btnCloseSettings = document.getElementById('btnCloseSettings');
const preferenceToggle = document.getElementById('preferenceToggle');
const btnResetPreferences = document.getElementById('btnResetPreferences');
const temperatureSlider = document.getElementById('temperatureSlider');
const maxLengthSlider = document.getElementById('maxLengthSlider');
const temperatureValue = document.getElementById('temperatureValue');
const maxLengthValue = document.getElementById('maxLengthValue');
const cpuPercent = document.getElementById('cpuPercent');
const ramPercent = document.getElementById('ramPercent');
const gpuPercent = document.getElementById('gpuPercent');
const btnCancel = document.getElementById('btnCancel');
const tabChat = document.getElementById('tabChat');
const tabImage = document.getElementById('tabImage');
const chatInputContainer = document.getElementById('chatInputContainer');
const imageInputContainer = document.getElementById('imageInputContainer');
const imagePromptInput = document.getElementById('imagePromptInput');
const btnGenerateImage = document.getElementById('btnGenerateImage');
const btnCancelImage = document.getElementById('btnCancelImage');
const btnUpload = document.getElementById('btnUpload');
const fileInput = document.getElementById('fileInput');
const uploadedFiles = document.getElementById('uploadedFiles');
const fileUploadArea = document.getElementById('fileUploadArea');
const btnMicrophone = document.getElementById('btnMicrophone');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    loadImageModels();
    loadConversations();
    loadStatus();
    loadPreferences();
    setupEventListeners();
    startSystemStatsUpdate();
});

// Event Listeners
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
    btnSettings.addEventListener('click', () => {
        settingsPanel.classList.add('open');
    });
    btnCloseSettings.addEventListener('click', () => {
        settingsPanel.classList.remove('open');
    });
    preferenceToggle.addEventListener('change', togglePreferenceLearning);
    btnResetPreferences.addEventListener('click', resetPreferences);
    temperatureSlider.addEventListener('input', (e) => {
        settings.temperature = parseFloat(e.target.value);
        temperatureValue.textContent = settings.temperature.toFixed(1);
    });
    maxLengthSlider.addEventListener('input', (e) => {
        settings.maxLength = parseInt(e.target.value);
        maxLengthValue.textContent = settings.maxLength;
    });
    tabChat.addEventListener('click', () => switchTab('chat'));
    tabImage.addEventListener('click', () => switchTab('image'));
    btnGenerateImage.addEventListener('click', generateImage);
    btnCancelImage.addEventListener('click', cancelImageGeneration);
    imagePromptInput.addEventListener('input', () => {
        btnGenerateImage.disabled = imagePromptInput.value.trim() === '';
    });
    btnUpload.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    
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
    btnMicrophone.addEventListener('click', toggleMicrophone);
}

function switchTab(tab) {
    currentTab = tab;
    if (tab === 'chat') {
        tabChat.classList.add('active');
        tabImage.classList.remove('active');
        chatInputContainer.style.display = 'block';
        imageInputContainer.style.display = 'none';
    } else {
        tabChat.classList.remove('active');
        tabImage.classList.add('active');
        chatInputContainer.style.display = 'none';
        imageInputContainer.style.display = 'block';
    }
}

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
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
        
        const response = await fetch(`${API_BASE}${endpoint}`, fetchOptions);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API Fehler');
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

async function loadModels() {
    try {
        const data = await apiCall('/models');
        modelSelect.innerHTML = '<option value="">Modell auswählen...</option>';
        
        // Nur Text-Modelle (keine Image-Modelle)
        for (const [id, model] of Object.entries(data.models)) {
            if (model.type !== 'image') {  // Überspringe Bildgenerierungsmodelle
                const option = document.createElement('option');
                option.value = id;
                option.textContent = model.name;
                if (id === data.current_model) {
                    option.selected = true;
                    currentModel = id;
                }
                modelSelect.appendChild(option);
            }
        }
        
        // Event-Listener nur einmal hinzufügen
        if (!modelSelect.hasAttribute('data-listener-added')) {
            modelSelect.addEventListener('change', async (e) => {
                const modelId = e.target.value;
                if (modelId) {
                    await loadModel(modelId);
                }
            });
            modelSelect.setAttribute('data-listener-added', 'true');
        }
    } catch (error) {
        console.error('Fehler beim Laden der Modelle:', error);
    }
}

async function loadImageModels() {
    try {
        const data = await apiCall('/image/models');
        imageModelSelect.innerHTML = '<option value="">Bildmodell auswählen...</option>';
        
        for (const [id, model] of Object.entries(data.models)) {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = model.name;
            if (id === data.current_model) {
                option.selected = true;
            }
            imageModelSelect.appendChild(option);
        }
        
        // Event-Listener für Bildmodell-Auswahl
        if (!imageModelSelect.hasAttribute('data-listener-added')) {
            imageModelSelect.addEventListener('change', async (e) => {
                const modelId = e.target.value;
                if (modelId) {
                    await loadImageModel(modelId);
                }
            });
            imageModelSelect.setAttribute('data-listener-added', 'true');
        }
    } catch (error) {
        console.error('Fehler beim Laden der Bildmodelle:', error);
        imageModelSelect.innerHTML = '<option value="">Fehler beim Laden</option>';
    }
}

async function loadImageModel(modelId) {
    try {
        statusText.textContent = 'Lade Bildmodell...';
        await apiCall('/image/models/load', {
            method: 'POST',
            body: JSON.stringify({ model_id: modelId })
        });
        statusText.textContent = 'Bildmodell geladen';
        setTimeout(() => {
            statusText.textContent = 'Bereit';
        }, 2000);
    } catch (error) {
        console.error('Fehler beim Laden des Bildmodells:', error);
        statusText.textContent = 'Fehler beim Laden';
        alert('Fehler beim Laden des Bildmodells: ' + error.message);
    }
}

async function loadModel(modelId) {
    try {
        statusText.textContent = 'Lade Modell...';
        await apiCall('/models/load', {
            method: 'POST',
            body: JSON.stringify({ model_id: modelId })
        });
        currentModel = modelId;
        statusText.textContent = 'Modell geladen';
        await loadStatus();
    } catch (error) {
        statusText.textContent = 'Fehler beim Laden';
        alert('Fehler beim Laden des Modells: ' + error.message);
    }
}

async function loadStatus() {
    try {
        const status = await apiCall('/status');
        if (status.model_loaded) {
            statusText.textContent = `Bereit (${status.current_model})`;
        } else {
            statusText.textContent = 'Kein Modell geladen';
        }
    } catch (error) {
        console.error('Fehler beim Laden des Status:', error);
    }
}

async function loadConversations() {
    try {
        const data = await apiCall('/conversations');
        renderConversations(data.conversations);
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
    
    conversationsList.innerHTML = conversations.map(conv => `
        <div class="conversation-item ${conv.id === currentConversationId ? 'active' : ''}" 
             data-id="${conv.id}">
            <span class="conversation-title">${escapeHtml(conv.title)}</span>
            <button class="conversation-delete" onclick="deleteConversation('${conv.id}', event)">×</button>
        </div>
    `).join('');
    
    // Add click listeners
    conversationsList.querySelectorAll('.conversation-item').forEach(item => {
        item.addEventListener('click', (e) => {
            if (!e.target.classList.contains('conversation-delete')) {
                loadConversation(item.dataset.id);
            }
        });
    });
}

async function loadConversation(conversationId) {
    try {
        const conversation = await apiCall(`/conversations/${conversationId}`);
        currentConversationId = conversationId;
        chatTitle.textContent = conversation.title;
        
        // Clear and render messages
        chatMessages.innerHTML = '';
        conversation.messages.forEach(msg => {
            addMessageToChat(msg.role, msg.content);
        });
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Update active conversation
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.toggle('active', item.dataset.id === conversationId);
        });
        
        loadConversations(); // Refresh list
    } catch (error) {
        console.error('Fehler beim Laden der Conversation:', error);
        alert('Fehler beim Laden der Conversation: ' + error.message);
    }
}

async function createNewConversation() {
    try {
        const data = await apiCall('/conversations', { method: 'POST' });
        currentConversationId = data.conversation_id;
        chatTitle.textContent = 'Neues Gespräch';
        chatMessages.innerHTML = '<div class="welcome-message"><h3>Neues Gespräch</h3><p>Stellen Sie eine Frage oder starten Sie ein Gespräch.</p></div>';
        loadConversations();
    } catch (error) {
        console.error('Fehler beim Erstellen der Conversation:', error);
        alert('Fehler beim Erstellen der Conversation: ' + error.message);
    }
}

async function deleteConversation(conversationId, event) {
    event.stopPropagation();
    if (!confirm('Gespräch wirklich löschen?')) return;
    
    try {
        await apiCall(`/conversations/${conversationId}`, { method: 'DELETE' });
        if (conversationId === currentConversationId) {
            createNewConversation();
        }
        loadConversations();
    } catch (error) {
        console.error('Fehler beim Löschen:', error);
        alert('Fehler beim Löschen: ' + error.message);
    }
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message && uploadedFilesList.length === 0) return;
    
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
        const response = await apiCall('/chat', {
            method: 'POST',
            body: JSON.stringify({
                message: fullMessage,
                conversation_id: currentConversationId,
                max_length: settings.maxLength,
                temperature: settings.temperature
            }),
            signal: currentAbortController.signal
        });
        
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
        addMessageToChat('assistant', response.response);
        
        // Update title if first message
        if (chatTitle.textContent === 'Neues Gespräch') {
            chatTitle.textContent = message.substring(0, 50) + (message.length > 50 ? '...' : '');
        }
        
    } catch (error) {
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
        // Reset UI
        btnSend.style.display = 'block';
        btnCancel.style.display = 'none';
        btnSend.disabled = messageInput.value.trim() === '';
        messageInput.disabled = false;
        currentAbortController = null;
    }
}

function cancelGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
        statusText.textContent = 'Abgebrochen';
    }
}

async function generateImage() {
    const prompt = imagePromptInput.value.trim();
    if (!prompt) return;
    
    // Prüfe ob ein Bildmodell ausgewählt ist
    const selectedModelId = imageModelSelect.value;
    if (!selectedModelId) {
        alert('Bitte wählen Sie zuerst ein Bildgenerierungsmodell aus!');
        return;
    }
    
    // Disable input
    imagePromptInput.disabled = true;
    btnGenerateImage.style.display = 'none';
    btnCancelImage.style.display = 'block';
    
    // Create AbortController
    currentImageAbortController = new AbortController();
    
    // Show loading message
    const loadingId = addMessageToChat('assistant', 'Generiere Bild...', true);
    statusText.textContent = 'Generiere Bild...';
    
    try {
        const response = await apiCall('/image/generate', {
            method: 'POST',
            body: JSON.stringify({
                prompt: prompt,
                negative_prompt: "",
                num_inference_steps: 20,
                guidance_scale: 7.5,
                width: 1024,
                height: 1024,
                model_id: selectedModelId  // Verwende ausgewähltes Modell
            }),
            signal: currentImageAbortController.signal
        });
        
        // Remove loading message
        const loadingMsg = document.getElementById(loadingId);
        if (loadingMsg) {
            loadingMsg.remove();
        }
        
        // Display image
        const imageUrl = `data:image/png;base64,${response.image_base64}`;
        addImageToChat(prompt, imageUrl);
        
        statusText.textContent = 'Bild generiert';
        
    } catch (error) {
        if (error.name === 'AbortError') {
            const loadingMsg = document.getElementById(loadingId);
            if (loadingMsg) {
                loadingMsg.remove();
            }
            addMessageToChat('assistant', 'Bildgenerierung abgebrochen.');
            statusText.textContent = 'Abgebrochen';
        } else {
            const loadingMsg = document.getElementById(loadingId);
            if (loadingMsg) {
                loadingMsg.remove();
            }
            addMessageToChat('assistant', 'Fehler: ' + error.message);
            console.error('Fehler bei Bildgenerierung:', error);
            statusText.textContent = 'Fehler';
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
        statusText.textContent = 'Abgebrochen';
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
    
    // Remove welcome message if exists
    const welcomeMsg = chatMessages.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
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
            ${escapeHtml(content)}
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
    
    return messageId;
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

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// System Stats Update
let statsUpdateInterval = null;
let lastStats = {
    cpu: null,
    ram: null,
    gpu: null
};

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
    // Aktualisiere alle 2 Sekunden
    statsUpdateInterval = setInterval(updateSystemStats, 2000);
}

async function updateSystemStats() {
    try {
        const stats = await apiCall('/system/stats', { method: 'GET' });
        
        if (!stats) {
            console.warn('Keine Stats-Daten erhalten');
            return;
        }
        
        console.log('System-Stats erhalten:', stats);
        
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
        console.error('Fehler beim Abrufen der System-Stats:', error);
        // Werte NICHT zurücksetzen - letzte Werte beibehalten
        // Nur wenn noch keine Werte vorhanden sind, "-" anzeigen
        if (cpuPercent && lastStats.cpu === null) cpuPercent.textContent = '-';
        if (ramPercent && lastStats.ram === null) ramPercent.textContent = '-';
        if (gpuPercent && lastStats.gpu === null) gpuPercent.textContent = '-';
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

// Auto-refresh status every 60 seconds (reduced frequency to avoid log spam)
setInterval(loadStatus, 60000);

// Audio Recording
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

async function toggleMicrophone() {
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
        statusText.textContent = 'Verarbeite Aufnahme...';
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
        
        statusText.textContent = 'Transkribiere...';
        
        const response = await fetch(`${API_BASE}/audio/transcribe`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Transkriptions-Fehler');
        }
        
        const data = await response.json();
        
        // Setze transkribierten Text in Input-Feld
        messageInput.value = data.text;
        btnSend.disabled = false;
        autoResizeTextarea();
        
        statusText.textContent = 'Transkription erfolgreich';
        setTimeout(() => {
            statusText.textContent = 'Bereit';
        }, 2000);
        
    } catch (error) {
        console.error('Fehler bei der Transkription:', error);
        alert('Fehler bei der Transkription: ' + error.message);
        statusText.textContent = 'Fehler';
        setTimeout(() => {
            statusText.textContent = 'Bereit';
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

