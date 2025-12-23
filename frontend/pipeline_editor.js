// Pipeline Editor - Node-System für Agent-Pipelines
let pipelineNodes = [];
let pipelineConnections = [];
let selectedNode = null;
let isDragging = false;
let dragOffset = { x: 0, y: 0 };

// DOM Elements
const pipelineEditorWindow = document.getElementById('pipelineEditorWindow');
const btnPipeline = document.getElementById('btnPipeline');
const btnClosePipelineEditor = document.getElementById('btnClosePipelineEditor');
const pipelineCanvas = document.getElementById('pipelineCanvas');
const btnAddPromptAgent = document.getElementById('btnAddPromptAgent');
const btnAddImageAgent = document.getElementById('btnAddImageAgent');
const btnAddVisionAgent = document.getElementById('btnAddVisionAgent');
const btnRunPipeline = document.getElementById('btnRunPipeline');
const pipelineInput = document.getElementById('pipelineInput');

// Initialize
if (btnPipeline) {
    btnPipeline.addEventListener('click', () => {
        pipelineEditorWindow.style.display = 'block';
        loadAgentTypes();
    });
}

if (btnClosePipelineEditor) {
    btnClosePipelineEditor.addEventListener('click', () => {
        pipelineEditorWindow.style.display = 'none';
    });
}

if (btnAddPromptAgent) {
    btnAddPromptAgent.addEventListener('click', () => addNode('prompt_agent', 'Prompt Agent'));
}

if (btnAddImageAgent) {
    btnAddImageAgent.addEventListener('click', () => addNode('image_agent', 'Image Agent'));
}

if (btnAddVisionAgent) {
    btnAddVisionAgent.addEventListener('click', () => addNode('vision_agent', 'Vision Agent'));
}

if (btnRunPipeline) {
    btnRunPipeline.addEventListener('click', executePipeline);
}

function addNode(agentType, agentName) {
    const node = {
        id: 'node_' + Date.now(),
        type: agentType,
        name: agentName,
        x: Math.random() * 300 + 50,
        y: Math.random() * 200 + 50,
        model_id: null
    };
    
    pipelineNodes.push(node);
    renderNodes();
}

function renderNodes() {
    pipelineCanvas.innerHTML = '';
    
    pipelineNodes.forEach(node => {
        const nodeElement = document.createElement('div');
        nodeElement.className = 'pipeline-node';
        nodeElement.id = node.id;
        nodeElement.style.left = node.x + 'px';
        nodeElement.style.top = node.y + 'px';
        nodeElement.innerHTML = `
            <div class="node-header">${escapeHtml(node.name)}</div>
            <div class="node-content">
                <select class="node-model-select" data-node-id="${node.id}">
                    <option value="">Default Modell</option>
                </select>
            </div>
            <div class="node-ports">
                <div class="port input-port" data-node-id="${node.id}">In</div>
                <div class="port output-port" data-node-id="${node.id}">Out</div>
            </div>
        `;
        
        pipelineCanvas.appendChild(nodeElement);
        
        // Event Listeners
        nodeElement.addEventListener('mousedown', (e) => startDrag(e, node));
        nodeElement.querySelector('.node-model-select').addEventListener('change', (e) => {
            node.model_id = e.target.value || null;
        });
        
        // Lade Modelle für Select
        loadModelsForNode(node.id, node.type);
    });
    
    renderConnections();
}

function startDrag(e, node) {
    if (e.target.classList.contains('port')) return;
    
    isDragging = true;
    selectedNode = node;
    dragOffset.x = e.clientX - node.x;
    dragOffset.y = e.clientY - node.y;
    
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', stopDrag);
}

function drag(e) {
    if (!isDragging || !selectedNode) return;
    
    selectedNode.x = e.clientX - dragOffset.x;
    selectedNode.y = e.clientY - dragOffset.y;
    
    const nodeElement = document.getElementById(selectedNode.id);
    if (nodeElement) {
        nodeElement.style.left = selectedNode.x + 'px';
        nodeElement.style.top = selectedNode.y + 'px';
    }
    
    renderConnections();
}

function stopDrag() {
    isDragging = false;
    selectedNode = null;
    document.removeEventListener('mousemove', drag);
    document.removeEventListener('mouseup', stopDrag);
}

function renderConnections() {
    // TODO: Implementiere Verbindungslinien zwischen Nodes
}

async function loadAgentTypes() {
    try {
        const response = await fetch(`${API_BASE}/agents/types`);
        const data = await response.json();
        // Agent-Typen sind bereits bekannt, keine weitere Aktion nötig
    } catch (error) {
        console.error('Fehler beim Laden der Agent-Typen:', error);
    }
}

async function loadModelsForNode(nodeId, agentType) {
    try {
        const response = await fetch(`${API_BASE}/models`);
        const data = await response.json();
        
        const select = document.querySelector(`.node-model-select[data-node-id="${nodeId}"]`);
        if (!select) return;
        
        // Filtere Modelle basierend auf Agent-Typ
        for (const [id, model] of Object.entries(data.models)) {
            if (agentType === 'image_agent' && model.type === 'image') {
                const option = document.createElement('option');
                option.value = id;
                option.textContent = model.name;
                select.appendChild(option);
            } else if (agentType !== 'image_agent' && model.type !== 'image' && model.type !== 'audio') {
                const option = document.createElement('option');
                option.value = id;
                option.textContent = model.name;
                select.appendChild(option);
            }
        }
    } catch (error) {
        console.error('Fehler beim Laden der Modelle:', error);
    }
}

async function executePipeline() {
    if (!currentConversationId) {
        alert('Bitte wählen Sie zuerst eine Conversation aus');
        return;
    }
    
    if (pipelineNodes.length === 0) {
        alert('Bitte fügen Sie mindestens einen Agenten hinzu');
        return;
    }
    
    const initialInput = pipelineInput.value.trim();
    if (!initialInput) {
        alert('Bitte geben Sie einen initialen Input ein');
        return;
    }
    
    // Speichere aktuelle Conversation-ID
    const conversationIdAtStart = currentConversationId;
    
    // Erstelle Pipeline-Steps
    const steps = pipelineNodes.map(node => ({
        agent_type: node.type,
        model_id: node.model_id || null
    }));
    
    try {
        btnRunPipeline.disabled = true;
        btnRunPipeline.textContent = 'Läuft...';
        
        const response = await fetch(`${API_BASE}/agents/pipeline?conversation_id=${conversationIdAtStart}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: 'Custom Pipeline',
                steps: steps,
                initial_input: initialInput
            })
        });
        
        const result = await response.json();
        
        // Prüfe ob User zu einer anderen Conversation gewechselt hat
        if (currentConversationId !== conversationIdAtStart) {
            console.log('User hat zu anderer Conversation gewechselt während Pipeline-Ausführung');
            return;
        }
        
        if (result.status === 'completed') {
            // Zeige Ergebnis in Chat (nicht als Alert, das blockiert)
            if (typeof addMessageToChat === 'function') {
                addMessageToChat('assistant', `Pipeline-Ergebnis: ${result.final_output}`);
            } else {
                alert(`Pipeline erfolgreich ausgeführt!\nFinaler Output: ${result.final_output.substring(0, 100)}...`);
            }
        } else {
            alert(`Pipeline-Fehler: ${result.status}`);
        }
    } catch (error) {
        // Prüfe ob User gewechselt hat
        if (currentConversationId !== conversationIdAtStart) {
            console.log('User hat zu anderer Conversation gewechselt während Pipeline-Ausführung');
            return;
        }
        
        console.error('Fehler bei Pipeline-Ausführung:', error);
        alert('Fehler bei Pipeline-Ausführung: ' + error.message);
    } finally {
        // Reset nur wenn noch in derselben Conversation
        if (currentConversationId === conversationIdAtStart) {
            btnRunPipeline.disabled = false;
            btnRunPipeline.textContent = '▶ Pipeline ausführen';
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

