// DOM Elements
        const activeContainer = document.getElementById('active-models-container');
        const libraryContainer = document.getElementById('library-container');
        const statsTableBody = document.getElementById('stats-table-body');
        const refreshToggle = document.getElementById('refresh-toggle');
        const metricsActive = document.getElementById('metrics-active');
        const metricsRam = document.getElementById('metrics-ram');
        const metricsRequests = document.getElementById('metrics-requests');

        // Tab Navigation
        function switchTab(tabName) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.textContent.toLowerCase().includes(tabName.substring(0,3)));
            if (activeBtn) activeBtn.classList.add('active');

            const content = document.getElementById(`tab-${tabName}`);
            if (content) content.classList.add('active');

            if (tabName === 'playground') {
                populatePlaygroundModelList();
            } else if (tabName === 'rag') {
                fetchRagIndexCatalog();
            }
        }

        let isRefreshing = true;
        let refreshInterval = null;

        function startAutoRefresh() {
            refreshInterval = setInterval(fetchData, 2000);
        }

        function stopAutoRefresh() {
            clearInterval(refreshInterval);
        }

        refreshToggle.addEventListener('change', (e) => {
            isRefreshing = e.target.checked;
            if (isRefreshing) {
                startAutoRefresh();
                fetchData();
            } else {
                stopAutoRefresh();
            }
        });

        // Initialize
        async function init() {
            await fetchData();
            startAutoRefresh();
            
            // Listen to temp range slider
            const slider = document.getElementById('chat-temp');
            const sliderVal = document.getElementById('chat-temp-val');
            slider.addEventListener('input', (e) => {
                sliderVal.textContent = e.target.value;
            });
        }

        // Fetch All API Data
        let globalLibraryData = null;
        let globalActiveData = null;

        async function fetchData() {
            try {
                const [activeRes, statsRes, libraryRes, historyRes] = await Promise.all([
                    fetch('/v1/models/active').then(r => r.json()),
                    fetch('/v1/models/stats').then(r => r.json()),
                    fetch('/v1/models').then(r => r.json()),
                    fetch('/v1/history?limit=50').then(r => r.json())
                ]);

                globalLibraryData = libraryRes;
                globalActiveData = activeRes;

                updateActiveModels(activeRes);
                updateStats(statsRes);
                updateHistory(historyRes);
                filterLibraryModels();
                updateSharedDropdowns(libraryRes);
                pollDownloadProgress();
                populatePlaygroundModelList();
            } catch (err) {
                console.error('Error fetching dashboard data:', err);
            }
        }

        function filterLibraryModels() {
            if (!globalLibraryData || !globalActiveData) return;
            
            const query = document.getElementById('library-search').value.toLowerCase().trim();
            const filteredData = globalLibraryData.data.filter(m => {
                return m.id.toLowerCase().includes(query);
            });
            
            const filteredLibrary = {
                data: filteredData
            };
            
            updateLibrary(filteredLibrary, globalActiveData);
        }

        // Update Active Model Cards
        function updateActiveModels(active) {
            metricsActive.textContent = active.length;
            
            let totalBytes = 0;
            active.forEach(a => totalBytes += (a.memory_bytes || 0));
            const totalGb = totalBytes / (1024 * 1024 * 1024);
            metricsRam.textContent = totalGb >= 1.0 ? `${totalGb.toFixed(2)} GB` : `${(totalBytes / (1024 * 1024)).toFixed(1)} MB`;

            if (active.length === 0) {
                activeContainer.innerHTML = `
                    <div style="grid-column: 1 / -1; width: 100%; border: 1px solid var(--card-border); background-color: var(--card-bg); border-radius: 20px; align-items: center; display: flex; justify-content: center; padding: 4rem; text-align: center; color: var(--text-secondary);">
                        <div>
                            <span style="font-size: 2.5rem; display: block; margin-bottom: 0.8rem;">💤</span>
                            <p>No model instances are currently running.<br>Use the Library below or select the model in your CLI to load them.</p>
                        </div>
                    </div>
                `;
                return;
            }

            activeContainer.innerHTML = active.map(a => {
                const typeClass = a.is_whisper ? 'badge-whisper' : (a.is_embedding ? 'badge-embedding' : 'badge-llm');
                const typeText = a.is_whisper ? 'Whisper' : (a.is_embedding ? 'Embedding' : 'LLM');
                const cpu = a.cpu_percent || 0.0;
                
                return `
                    <div class="active-card">
                        <div class="active-header">
                            <h3 class="active-model-name">${a.model}</h3>
                            <span class="active-badge ${typeClass}">${typeText}</span>
                        </div>
                        
                        <div class="active-details">
                            <div class="detail-item">
                                <span>Gateway Port</span>
                                <p>${a.port}</p>
                            </div>
                            <div class="detail-item">
                                <span>Idle Time</span>
                                <p>${a.idle_seconds}s</p>
                            </div>
                        </div>

                        <div class="active-resources">
                            <div class="resource-bar-wrapper">
                                <div class="resource-bar-label">
                                    <span>CPU Usage</span>
                                    <span>${cpu.toFixed(1)}%</span>
                                </div>
                                <div class="resource-bar">
                                    <div class="resource-bar-fill" style="width: ${Math.min(cpu, 100)}%;"></div>
                                </div>
                            </div>

                            <div class="resource-bar-wrapper">
                                <div class="resource-bar-label">
                                    <span>RAM Footprint</span>
                                    <span>${a.memory_str}</span>
                                </div>
                                <div class="resource-bar">
                                    <div class="resource-bar-fill" style="width: ${Math.min((a.memory_bytes || 0) / (8 * 1024 * 1024 * 1024) * 100, 100)}%; background: linear-gradient(90deg, var(--accent-violet), var(--accent-rose));"></div>
                                </div>
                            </div>
                        </div>

                        <div class="active-actions">
                            <button class="btn btn-stop" onclick="stopModel('${a.model}')">
                                <span>⏹</span> Stop Model
                            </button>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Update Cumulative Statistics Table
        function updateStats(stats) {
            let totalReqs = 0;
            const rows = Object.entries(stats).map(([model, data]) => {
                totalReqs += (data.request_count || 0);
                const avgSpeed = data.avg_speed_tps > 0 ? `${data.avg_speed_tps.toFixed(1)} tok/s` : 'N/A';
                const errors = data.error_count || 0;
                
                return `
                    <tr>
                        <td style="font-weight: 700; font-family: monospace; font-size: 0.8rem; word-break: break-all;">${model}</td>
                        <td>${data.request_count} ${errors > 0 ? `<span style="color: var(--accent-rose); font-size: 0.75rem;">(${errors} err)</span>` : ''}</td>
                        <td>${data.prompt_tokens.toLocaleString()}</td>
                        <td>${data.completion_tokens.toLocaleString()}</td>
                        <td>${(data.avg_latency_ms / 1000).toFixed(2)}s</td>
                        <td style="color: var(--accent-emerald); font-weight: 600;">${avgSpeed}</td>
                    </tr>
                `;
            }).join('');

            metricsRequests.textContent = totalReqs;
            
            if (rows.length === 0) {
                statsTableBody.innerHTML = `
                    <tr>
                        <td colspan="6" style="text-align: center; color: var(--text-secondary); padding: 3rem;">No request statistics recorded yet.</td>
                    </tr>
                `;
            } else {
                statsTableBody.innerHTML = rows;
            }
        }

        // Update Request History Log Table
        function updateHistory(history) {
            const historyTableBody = document.getElementById('history-table-body');
            if (!historyTableBody) return;

            if (!history || history.length === 0) {
                historyTableBody.innerHTML = `
                    <tr>
                        <td colspan="7" style="text-align: center; color: var(--text-secondary); padding: 2rem;">No request history recorded yet.</td>
                    </tr>
                `;
                return;
            }

            const rows = history.map(item => {
                const statusBadge = item.is_error 
                    ? `<span style="color: var(--accent-rose); font-weight: 700; font-size: 0.75rem;">ERROR</span>`
                    : `<span style="color: var(--accent-emerald); font-weight: 700; font-size: 0.75rem;">OK</span>`;
                const snippet = item.prompt_snippet ? (item.prompt_snippet.length > 35 ? item.prompt_snippet.substring(0, 32) + '...' : item.prompt_snippet) : '-';
                const tokens = `${item.prompt_tokens || 0} / ${item.completion_tokens || 0}`;
                const duration = item.duration_sec > 0 ? `${item.duration_sec.toFixed(2)}s` : '-';

                return `
                    <tr style="cursor: pointer; transition: background-color 0.2s;" onclick="inspectHistoryRecord(${item.id})" title="Click to inspect full sent & received messages">
                        <td style="color: var(--text-secondary); font-size: 0.8rem;">${item.timestamp}</td>
                        <td style="font-weight: 600; font-family: monospace; font-size: 0.8rem;">${item.model_name}</td>
                        <td style="color: var(--accent-cyan); font-size: 0.8rem;">${item.endpoint}</td>
                        <td style="font-size: 0.8rem; color: var(--text-primary); max-width: 200px; word-break: break-all;">${snippet}</td>
                        <td style="font-size: 0.8rem;">${tokens}</td>
                        <td style="font-size: 0.8rem; color: var(--accent-amber);">${duration}</td>
                        <td>${statusBadge}</td>
                    </tr>
                `;
            }).join('');

            historyTableBody.innerHTML = rows;
        }

        async function inspectHistoryRecord(id) {
            try {
                const res = await fetch(`/v1/history/${id}`);
                if (!res.ok) return;
                const data = await res.json();

                document.getElementById('modal-title').textContent = `🔍 Request Inspection #${data.id}`;
                document.getElementById('modal-meta').innerHTML = `
                    <div><span style="color: var(--text-secondary);">Timestamp:</span> ${data.timestamp}</div>
                    <div><span style="color: var(--text-secondary);">Model:</span> <span style="color: var(--accent-cyan); font-weight: 600;">${data.model_name}</span></div>
                    <div><span style="color: var(--text-secondary);">Endpoint:</span> ${data.endpoint}</div>
                    <div><span style="color: var(--text-secondary);">Tokens:</span> ${data.prompt_tokens || 0} P / ${data.completion_tokens || 0} C</div>
                    <div><span style="color: var(--text-secondary);">Duration:</span> ${data.duration_sec || 0}s</div>
                `;

                const promptElem = document.getElementById('modal-prompt-content');
                const respElem = document.getElementById('modal-response-content');

                promptElem.textContent = typeof data.full_prompt === 'object' ? JSON.stringify(data.full_prompt, null, 2) : (data.full_prompt || '-');
                respElem.textContent = typeof data.full_response === 'object' ? JSON.stringify(data.full_response, null, 2) : (data.full_response || '-');

                document.getElementById('history-modal').style.display = 'flex';
            } catch (err) {
                console.error('Error inspecting request record:', err);
            }
        }

        function closeHistoryModal() {
            document.getElementById('history-modal').style.display = 'none';
        }



        // Update Library Model List
        function updateLibrary(library, active) {
            const activeModels = new Set(active.map(a => a.model));
            
            if (!library.data || library.data.length === 0) {
                libraryContainer.innerHTML = `
                    <div style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                        <p>No local models found on disk.<br>Use the Model Hub tab to pull models.</p>
                    </div>
                `;
                return;
            }

            libraryContainer.innerHTML = library.data.map((m, idx) => {
                const isRunning = activeModels.has(m.id);
                const isWhisper = m.id.toLowerCase().includes('whisper');
                const isEmbedding = m.id.toLowerCase().includes('embed') || m.id.toLowerCase().includes('bert');
                
                let selectedType = 'llm';
                if (isWhisper) selectedType = 'whisper';
                else if (isEmbedding) selectedType = 'embedding';

                return `
                    <div class="library-item">
                        <div class="library-item-top">
                            <span class="library-name">${m.id}</span>
                            <span class="library-size">Local File</span>
                        </div>
                        
                        <div class="library-controls">
                            <div class="library-params">
                                <div class="param-group">
                                    <label>Type:</label>
                                    <select id="type-${idx}" ${isRunning ? 'disabled' : ''}>
                                        <option value="llm" ${selectedType === 'llm' ? 'selected' : ''}>LLM</option>
                                        <option value="embedding" ${selectedType === 'embedding' ? 'selected' : ''}>Embedding</option>
                                        <option value="whisper" ${selectedType === 'whisper' ? 'selected' : ''}>Whisper</option>
                                    </select>
                                </div>
                                <div class="param-group">
                                    <label>Timeout (s):</label>
                                    <input type="number" id="timeout-${idx}" value="300" min="0" ${isRunning ? 'disabled' : ''}>
                                </div>
                            </div>
                            
                            <div class="library-actions" style="display: flex; gap: 0.5rem;">
                                ${isRunning ? `
                                    <button class="btn btn-stop" onclick="stopModel('${m.id}')">
                                        ⏹ Stop
                                    </button>
                                ` : `
                                    <button class="btn btn-load" onclick="loadModel('${m.id}', ${idx})">
                                        ▶ Load
                                    </button>
                                    <button class="btn btn-stop" onclick="deleteModel('${m.id}')" title="Delete model from disk" style="padding: 0.6rem 0.8rem; background-color: var(--accent-rose); border-color: var(--accent-rose);">
                                        🗑️
                                    </button>
                                `}
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Shared dropdown updates
        function updateSharedDropdowns(library) {
            const indexModelSelect = document.getElementById('index-model-select');
            const searchModelSelect = document.getElementById('rag-search-model-select');
            if (!indexModelSelect || !library.data) return;

            const prevIndexVal = indexModelSelect.value;
            const prevSearchVal = searchModelSelect.value;

            // Filter local models containing 'embed' or 'bert'
            const embModels = library.data.filter(m => m.id.toLowerCase().includes('embed') || m.id.toLowerCase().includes('bert'));

            if (embModels.length === 0) {
                const opt = `<option value="">No local embedding models found</option>`;
                indexModelSelect.innerHTML = opt;
                searchModelSelect.innerHTML = opt;
            } else {
                const options = embModels.map(m => `<option value="${m.id}">${m.id}</option>`).join('');
                indexModelSelect.innerHTML = options;
                searchModelSelect.innerHTML = `<option value="">(Auto-detect from database)</option>` + options;

                if (prevIndexVal) indexModelSelect.value = prevIndexVal;
                if (prevSearchVal) searchModelSelect.value = prevSearchVal;
            }
        }

        // Stop Model API call
        async function stopModel(modelName) {
            try {
                const res = await fetch('/v1/models/unload', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: modelName })
                });
                if (res.ok) {
                    fetchData();
                    populatePlaygroundModelList();
                } else {
                    const data = await res.json();
                    alert(`Error unloading: ${data.error || res.statusText}`);
                }
            } catch (err) {
                alert(`Error stopping model: ${err}`);
            }
        }

        async function deleteModel(modelId) {
            const confirmDel = window.confirm(`Are you sure you want to delete '${modelId}' from your local disk?`);
            if (!confirmDel) return;

            try {
                const res = await fetch('/v1/models/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: modelId })
                }).then(r => r.json());

                if (res.error) {
                    alert(`Failed to delete model: ${res.error}`);
                } else {
                    alert(`Successfully deleted model: ${modelId}`);
                    fetchData();
                    populatePlaygroundModelList();
                }
            } catch (e) {
                alert(`Error deleting model: ${e}`);
            }
        }

        // Load Model API call
        async function loadModel(modelName, idx) {
            const type = document.getElementById(`type-${idx}`).value;
            const timeoutVal = parseInt(document.getElementById(`timeout-${idx}`).value, 10);
            
            const isWhisper = type === 'whisper';
            const isEmbedding = type === 'embedding';

            try {
                const btn = document.querySelector(`[onclick="loadModel('${modelName}', ${idx})"]`);
                if (btn) {
                    btn.disabled = true;
                    btn.textContent = '⏳ Loading...';
                }

                const res = await fetch('/v1/models/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: modelName,
                        is_whisper: isWhisper,
                        is_embedding: isEmbedding,
                        idle_timeout: timeoutVal
                    })
                });
                
                if (res.ok) {
                    fetchData();
                    populatePlaygroundModelList();
                } else {
                    const data = await res.json();
                    alert(`Error loading: ${data.error || res.statusText}`);
                    fetchData();
                }
            } catch (err) {
                alert(`Error loading model: ${err}`);
                fetchData();
            }
        }

        /* Playground JavaScript */
        /* Playground JavaScript */
        const SYSTEM_PRESETS = {
            default: "You are a helpful coding assistant. Answer concisely.",
            developer: "You are an expert software engineer. Provide high-quality, clean, well-commented code. Focus on edge cases and optimal performance.",
            academic: "You are an academic researcher and tutor. Explain complex concepts in an educational, structured, and detailed manner with references and logical breakdowns.",
            creative: "You are a creative writer. Help the user brainstorm ideas, write stories, draft poetry, and express ideas in an engaging and expressive style."
        };

        function applySystemPreset() {
            const select = document.getElementById('chat-system-preset');
            const textarea = document.getElementById('chat-system-prompt');
            if (select.value && SYSTEM_PRESETS[select.value]) {
                textarea.value = SYSTEM_PRESETS[select.value];
            }
        }

        async function populatePlaygroundModelList() {
            const select = document.getElementById('chat-model-select');
            const whisperSelect = document.getElementById('chat-whisper-select');
            try {
                const [allModelsData, active] = await Promise.all([
                    fetch('/v1/models').then(r => r.json()),
                    fetch('/v1/models/active').then(r => r.json())
                ]);
                
                const downloaded = allModelsData.data || [];
                const activeSet = new Set(active.map(a => a.model));
                const prevVal = select.value;
                const prevWhisperVal = whisperSelect.value;

                // Separate LLMs and Whisper models based on id names
                const llms = [];
                const whispers = [];

                downloaded.forEach(m => {
                    const id = m.id;
                    const isWhisper = id.toLowerCase().includes('whisper') || id.endsWith('.bin');
                    if (isWhisper) {
                        whispers.push(m);
                    } else {
                        llms.push(m);
                    }
                });
                
                if (llms.length === 0) {
                    select.innerHTML = `<option value="">No Downloaded LLMs Available</option>`;
                    document.getElementById('chat-model-indicator').textContent = 'Select an LLM on the left';
                    document.getElementById('chat-status-indicator').textContent = 'Playground Sandbox';
                } else {
                    select.innerHTML = llms.map(l => {
                        const isActive = activeSet.has(l.id);
                        const label = isActive ? `🟢 ${l.id} (Active)` : `⚪ ${l.id}`;
                        return `<option value="${l.id}">${label}</option>`;
                    }).join('');
                    
                    if (prevVal && llms.some(l => l.id === prevVal)) {
                        select.value = prevVal;
                    } else {
                        select.value = llms[0].id;
                    }
                    updatePlaygroundHeader();
                }

                if (whispers.length === 0) {
                    whisperSelect.innerHTML = `<option value="">No Downloaded Whisper Models</option>`;
                } else {
                    whisperSelect.innerHTML = whispers.map(w => {
                        const isActive = activeSet.has(w.id);
                        const label = isActive ? `🟢 ${w.id} (Active)` : `⚪ ${w.id}`;
                        return `<option value="${w.id}">${label}</option>`;
                    }).join('');
                    
                    if (prevWhisperVal && whispers.some(w => w.id === prevWhisperVal)) {
                        whisperSelect.value = prevWhisperVal;
                    } else {
                        whisperSelect.value = whispers[0].id;
                    }
                }
            } catch(e) {
                console.error(e);
            }
        }

        function updatePlaygroundHeader() {
            const select = document.getElementById('chat-model-select');
            const header = document.getElementById('chat-model-indicator');
            if (select.value) {
                const optionText = select.options[select.selectedIndex]?.text || '';
                const isActive = optionText.includes('🟢');
                header.textContent = select.value;
                document.getElementById('chat-status-indicator').textContent = isActive ? 'Online & Ready' : 'Inactive (Will auto-load on send)';
            } else {
                header.textContent = 'Select an LLM on the left';
                document.getElementById('chat-status-indicator').textContent = 'Playground Sandbox';
            }
        }

        document.getElementById('chat-model-select').addEventListener('change', updatePlaygroundHeader);

        let playgroundChatMessages = [];

        function clearPlaygroundChat() {
            playgroundChatMessages = [];
            document.getElementById('chat-history-container').innerHTML = `
                <div class="bubble-system">Session history cleared. Send a message to start a new chat.</div>
            `;
        }

        function handleChatSubmit(e) {
            if (e.key === 'Enter') {
                sendPlaygroundMessage();
            }
        }

        let currentImageBase64 = null;

        function handleChatImageUpload(e) {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(evt) {
                currentImageBase64 = evt.target.result;
                document.getElementById('chat-image-preview').src = currentImageBase64;
                document.getElementById('chat-image-preview-container').style.display = 'flex';
            };
            reader.readAsDataURL(file);
        }

        function clearChatImageAttachment() {
            currentImageBase64 = null;
            document.getElementById('chat-file-input').value = '';
            document.getElementById('chat-image-preview').src = '';
            document.getElementById('chat-image-preview-container').style.display = 'none';
        }

        function escapeHtml(text) {
            if (!text) return '';
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        function parseChatResponse(reasoningEnabled, fullText) {
            if (fullText.includes('<think>')) {
                const parts = fullText.split('<think>');
                const beforeThink = parts[0];
                const rest = parts.slice(1).join('<think>');

                if (rest.includes('</think>')) {
                    const subparts = rest.split('</think>');
                    const thinking = subparts[0];
                    const afterThink = subparts.slice(1).join('</think>');

                    return {
                        thinking: reasoningEnabled ? thinking : '',
                        answer: beforeThink + afterThink
                    };
                } else {
                    return {
                        thinking: reasoningEnabled ? rest : '',
                        answer: beforeThink
                    };
                }
            }
            return {
                thinking: '',
                answer: fullText
            };
        }

        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;

        async function toggleVoiceRecording() {
            const micBtn = document.getElementById('chat-mic-btn');
            const whisperSelect = document.getElementById('chat-whisper-select');
            const whisperModel = whisperSelect.value;

            if (!whisperModel) {
                alert("Please select/download a Whisper model first for voice input.");
                return;
            }

            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioChunks = [];
                    mediaRecorder = new MediaRecorder(stream);
                    
                    mediaRecorder.ondataavailable = e => {
                        if (e.data.size > 0) {
                            audioChunks.push(e.data);
                        }
                    };

                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        stream.getTracks().forEach(track => track.stop());
                        await transcribeAudioBlob(audioBlob, whisperModel);
                    };

                    mediaRecorder.start();
                    isRecording = true;
                    micBtn.textContent = '🛑';
                    micBtn.style.background = 'rgba(244, 63, 94, 0.2)';
                    micBtn.style.color = 'var(--accent-rose)';
                    micBtn.style.borderColor = 'var(--accent-rose)';
                    micBtn.title = 'Stop Recording';
                } catch (err) {
                    alert(`Microphone access failed: ${err}`);
                }
            } else {
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                }
                isRecording = false;
                micBtn.textContent = '🎙️';
                micBtn.style.background = '';
                micBtn.style.color = '';
                micBtn.style.borderColor = '';
                micBtn.title = 'Voice Input (Whisper)';
            }
        }

        async function transcribeAudioBlob(blob, whisperModel) {
            const statusIndicator = document.getElementById('chat-status-indicator');
            const originalStatus = statusIndicator.textContent;
            statusIndicator.textContent = 'Transcribing voice input...';
            
            const formData = new FormData();
            formData.append('file', blob, 'recording.wav');
            formData.append('model', whisperModel);
            
            try {
                const res = await fetch('/v1/audio/transcriptions', {
                    method: 'POST',
                    body: formData
                });
                
                if (!res.ok) {
                    const data = await res.json();
                    alert(`Transcription failed: ${data.error || res.statusText}`);
                    return;
                }
                
                const data = await res.json();
                const text = data.text || '';
                if (text.trim()) {
                    const input = document.getElementById('chat-user-input');
                    input.value = (input.value ? input.value + ' ' : '') + text.trim();
                }
            } catch (err) {
                alert(`Error transcribing audio: ${err}`);
            } finally {
                statusIndicator.textContent = originalStatus;
            }
        }

        async function sendPlaygroundMessage() {
            const select = document.getElementById('chat-model-select');
            const model = select.value;
            if (!model) {
                alert("Please download or select an LLM model first.");
                return;
            }

            const input = document.getElementById('chat-user-input');
            const message = input.value.trim();
            if (!message && !currentImageBase64) return;

            input.value = '';

            const container = document.getElementById('chat-history-container');
            
            // Append User message
            let userBubbleContent = '';
            if (currentImageBase64) {
                userBubbleContent += `<img src="${currentImageBase64}" style="max-width: 100%; max-height: 200px; display: block; border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid var(--card-border);">`;
            }
            if (message) {
                userBubbleContent += `<span>${escapeHtml(message)}</span>`;
            } else {
                userBubbleContent += `<span style="font-style: italic; color: rgba(255,255,255,0.7);">[Attached Image]</span>`;
            }
            
            container.innerHTML += `<div class="chat-bubble bubble-user">${userBubbleContent}</div>`;
            container.scrollTop = container.scrollHeight;

            const systemPrompt = document.getElementById('chat-system-prompt').value.trim();
            const temp = parseFloat(document.getElementById('chat-temp').value);

            // Construct payload messages
            const messages = [];
            if (systemPrompt) {
                messages.push({ role: 'system', content: systemPrompt });
            }
            
            let userMsgContent;
            if (currentImageBase64) {
                userMsgContent = [
                    { type: 'text', text: message || "Describe the image." },
                    { type: 'image_url', image_url: { url: currentImageBase64 } }
                ];
            } else {
                userMsgContent = message;
            }

            playgroundChatMessages.push({ role: 'user', content: userMsgContent });
            
            clearChatImageAttachment();

            playgroundChatMessages.forEach(m => messages.push(m));

            const assistantBubbleId = 'assistant-bubble-' + Date.now();
            const isModelRunning = select.options[select.selectedIndex]?.text.includes('🟢');
            const loadingText = isModelRunning ? '⏳ ...' : '⏳ Model is inactive. Loading weights into memory (this may take up to 30 seconds)...';
            
            container.innerHTML += `<div class="chat-bubble bubble-assistant" id="${assistantBubbleId}">${loadingText}</div>`;
            container.scrollTop = container.scrollHeight;

            const bubbleElement = document.getElementById(assistantBubbleId);
            const reasoningEnabled = document.getElementById('chat-reasoning').checked;
            
            const startTime = Date.now();
            try {
                const res = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: model,
                        messages: messages,
                        temperature: temp,
                        stream: true,
                        chat_template_kwargs: {
                            enable_thinking: reasoningEnabled
                        }
                    })
                });

                if (!res.ok) {
                    const data = await res.json();
                    bubbleElement.innerHTML = `<span style="color: var(--accent-rose);">Error: ${data.error || res.statusText}</span>`;
                    return;
                }

                if (!isModelRunning) {
                    populatePlaygroundModelList();
                }

                bubbleElement.innerHTML = '';
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let assistantAnswer = '';
                let assistantReasoning = '';

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (trimmed.startsWith('data: ')) {
                            const dataStr = trimmed.slice(6).trim();
                            if (dataStr === '[DONE]') break;
                            try {
                                const data = JSON.parse(dataStr);
                                const contentDelta = data.choices[0].delta.content || '';
                                const reasoningDelta = data.choices[0].delta.reasoning_content || '';
                                
                                assistantAnswer += contentDelta;
                                assistantReasoning += reasoningDelta;
                                
                                const reasoningEnabled = document.getElementById('chat-reasoning').checked;
                                
                                let displayThinking = '';
                                let displayAnswer = '';
                                
                                if (assistantReasoning) {
                                    if (reasoningEnabled) {
                                        displayThinking = assistantReasoning;
                                    }
                                    displayAnswer = assistantAnswer;
                                } else {
                                    const parsed = parseChatResponse(reasoningEnabled, assistantAnswer);
                                    displayThinking = parsed.thinking;
                                    displayAnswer = parsed.answer;
                                }
                                
                                let bubbleHtml = '';
                                if (displayThinking) {
                                    bubbleHtml += `
                                        <div class="thinking-block">
                                            <div class="thinking-header">💭 Thinking Process...</div>
                                            <div class="thinking-content">${escapeHtml(displayThinking)}</div>
                                        </div>
                                    `;
                                }
                                
                                if (typeof marked !== 'undefined') {
                                    bubbleHtml += `<div class="markdown-body">${marked.parse(displayAnswer)}</div>`;
                                } else {
                                    bubbleHtml += `<span>${escapeHtml(displayAnswer)}</span>`;
                                }
                                
                                bubbleElement.innerHTML = bubbleHtml;
                                if (typeof Prism !== 'undefined') {
                                    Prism.highlightAllUnder(bubbleElement);
                                }
                                container.scrollTop = container.scrollHeight;
                            } catch(e) {}
                        }
                    }
                }
                
                const endTime = Date.now();
                const durationSec = (endTime - startTime) / 1000;
                const completionWords = assistantAnswer.split(/\s+/).length;
                const estTokens = Math.round(completionWords * 1.3);
                const tokensPerSec = (estTokens / durationSec).toFixed(1);
                
                const statsHtml = `
                    <span class="chat-stats">
                        ⏱️ ${durationSec.toFixed(2)}s | ⚡ ${tokensPerSec} tok/sec (est. ${estTokens} tokens)
                    </span>
                `;
                const statsSpan = document.createElement('div');
                statsSpan.innerHTML = statsHtml;
                bubbleElement.appendChild(statsSpan);
                container.scrollTop = container.scrollHeight;

                playgroundChatMessages.push({ role: 'assistant', content: assistantAnswer });
            } catch (err) {
                bubbleElement.innerHTML = `<span style="color: var(--accent-rose);">Connection error: ${err}</span>`;
            }
        }

        /* Model Hub Downloader JS */
        function handleHubSearch(e) {
            if (e.key === 'Enter') {
                searchHuggingFaceHub();
            }
        }

        const loadedModelFiles = {};

        async function loadModelFiles(repo) {
            const selectId = `quant-select-${repo.replace(/\//g, '-')}`;
            const select = document.getElementById(selectId);
            if (!select || loadedModelFiles[repo]) return;

            loadedModelFiles[repo] = true;
            select.innerHTML = '<option value="">⏳ Loading files...</option>';

            try {
                const files = await fetch(`/v1/hf/files?model=${encodeURIComponent(repo)}`).then(r => r.json());
                if (files.error || !Array.isArray(files) || files.length === 0) {
                    select.innerHTML = '<option value="">No GGUF files found</option>';
                    return;
                }

                select.innerHTML = files.map(f => `<option value="${f}">${f}</option>`).join('');
            } catch (e) {
                select.innerHTML = '<option value="">Error loading files</option>';
                delete loadedModelFiles[repo];
            }
        }

        async function searchHuggingFaceHub() {
            const query = document.getElementById('hf-search-query').value.trim();
            if (!query) return;

            const tbody = document.getElementById('hub-search-results');
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align: center; color: var(--text-secondary); padding: 3rem;">🔍 Searching Hugging Face Hub...</td>
                </tr>
            `;

            try {
                const results = await fetch(`/v1/hf/search?query=${query}`).then(r => r.json());
                if (results.error) {
                    tbody.innerHTML = `
                        <tr>
                            <td colspan="6" style="text-align: center; color: var(--accent-rose); padding: 3rem;">Search failed: ${results.error}</td>
                        </tr>
                    `;
                    return;
                }

                if (results.length === 0) {
                    tbody.innerHTML = `
                        <tr>
                            <td colspan="6" style="text-align: center; color: var(--text-secondary); padding: 3rem;">No GGUF models matched query.</td>
                        </tr>
                    `;
                    return;
                }

                tbody.innerHTML = results.map(r => {
                    const repo = r.id;
                    const author = r.author || 'Unknown';
                    const dl = r.downloads || 0;
                    const likes = r.likes || 0;
                    const dlStr = dl >= 1000000 ? `${(dl/1000000).toFixed(1)}M` : (dl >= 1000 ? `${(dl/1000).toFixed(1)}k` : dl);
                    const safeRepoId = repo.replace(/\//g, '-');

                    return `
                        <tr>
                            <td style="font-weight: 700; font-family: monospace; font-size: 0.85rem;">${repo}</td>
                            <td>${author}</td>
                            <td>${dlStr}</td>
                            <td>❤️ ${likes}</td>
                            <td>
                                <select id="quant-select-${safeRepoId}" class="input-text" style="width: 250px; font-size: 0.75rem; padding: 0.35rem 0.6rem; border-radius: 6px; border: 1px solid var(--card-border);" onclick="loadModelFiles('${repo}')">
                                    <option value="">(Click to load files...)</option>
                                </select>
                            </td>
                            <td>
                                <button class="btn btn-secondary" onclick="pullModel('${repo}')" style="padding: 0.4rem 0.8rem; font-size: 0.75rem;">
                                    ⬇ Pull Model
                                </button>
                            </td>
                        </tr>
                    `;
                }).join('');

            } catch (err) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" style="text-align: center; color: var(--accent-rose); padding: 3rem;">Error connecting to gateway: ${err}</td>
                    </tr>
                `;
            }
        }

        async function pullModel(repoId) {
            const selectId = `quant-select-${repoId.replace(/\//g, '-')}`;
            const select = document.getElementById(selectId);
            const chosenFile = select ? select.value : '';

            if (!chosenFile) {
                alert("Please select a specific quantization file first! (Click the dropdown next to the model to load files).");
                return;
            }

            const fullModelId = `${repoId}:${chosenFile}`;

            try {
                const res = await fetch('/v1/models/pull', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: fullModelId })
                }).then(r => r.json());

                if (res.status === 'started') {
                    alert(`Download started for model: ${fullModelId}\nCheck the progress bars above.`);
                    pollDownloadProgress();
                } else if (res.status === 'already_pulling') {
                    alert("This model is already currently downloading.");
                } else {
                    alert(`Failed to start: ${res.error || 'Unknown error'}`);
                }
            } catch(e) {
                alert(`Error: ${e}`);
            }
        }

        async function pollDownloadProgress() {
            const container = document.getElementById('download-tasks-container');
            try {
                const tasks = await fetch('/v1/models/pull/status').then(r => r.json());
                const entries = Object.entries(tasks).filter(([name, data]) => data.status === 'downloading' || data.status === 'pending');
                
                if (entries.length === 0) {
                    container.innerHTML = '';
                    return;
                }

                container.innerHTML = entries.map(([name, data]) => {
                    const progress = data.progress || 0;
                    return `
                        <div class="download-task-card">
                            <div class="download-task-info">
                                <span class="download-task-name">Downloading: ${name}</span>
                                <span class="download-task-pct">${progress}%</span>
                            </div>
                            <div class="progress-track">
                                <div class="progress-bar-fill" style="width: ${progress}%;"></div>
                            </div>
                        </div>
                    `;
                }).join('');

                // Schedule next status fetch in 1 second
                setTimeout(pollDownloadProgress, 1000);

            } catch(e) {
                console.error('Error polling pull status:', e);
                // Retry in 2 seconds on error
                setTimeout(pollDownloadProgress, 2000);
            }
        }

        /* RAG & Indexer JavaScript */
        function handleProjectDirChange() {
            const projectDir = document.getElementById('rag-project-dir').value.trim();
            const indexDirInput = document.getElementById('index-dir-path');
            if (indexDirInput && projectDir) {
                indexDirInput.value = projectDir;
            }
            fetchRagIndexCatalog();
        }

        async function fetchRagIndexCatalog() {
            const container = document.getElementById('rag-indexed-list');
            const projectDir = document.getElementById('rag-project-dir').value.trim();
            try {
                let url = '/v1/db/list';
                if (projectDir) {
                    url += `?directory=${encodeURIComponent(projectDir)}`;
                }
                const list = await fetch(url).then(r => r.json());
                
                if (list.length === 0) {
                    container.innerHTML = `
                        <div style="text-align: center; color: var(--text-secondary); padding: 3rem;">
                            <p>No documents or folders currently indexed ${projectDir ? 'in this directory' : 'globally'}.</p>
                        </div>
                    `;
                    return;
                }

                container.innerHTML = list.map(doc => {
                    return `
                        <div class="source-card">
                            <div class="source-info">
                                <h4>${doc.file_path}</h4>
                                <p>Model: ${doc.model_name} | Chunks: <strong>${doc.chunks}</strong></p>
                            </div>
                            <button class="btn btn-stop" onclick="removeIndexedDoc('${doc.file_path}')" style="padding: 0.4rem 0.8rem; font-size: 0.75rem;">
                                Remove
                            </button>
                        </div>
                    `;
                }).join('');

            } catch(e) {
                console.error(e);
            }
        }

        async function removeIndexedDoc(filePath) {
            const confirm = window.confirm(`Are you sure you want to remove '${filePath}' from the vector database?`);
            if (!confirm) return;

            try {
                const res = await fetch('/v1/db/remove', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: filePath })
                }).then(r => r.json());

                if (res.status === 'removed') {
                    fetchRagIndexCatalog();
                } else {
                    alert(`Failed to remove: ${res.error}`);
                }
            } catch(e) {
                alert(`Error removing document: ${e}`);
            }
        }

        async function indexLocalFolder() {
            const dir = document.getElementById('index-dir-path').value.trim();
            const select = document.getElementById('index-model-select');
            const model = select.value;

            if (!dir) {
                alert("Please enter a directory path.");
                return;
            }
            if (!model) {
                alert("Please select a local embedding model.");
                return;
            }

            try {
                const res = await fetch('/v1/db/index', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ directory: dir, model: model })
                }).then(r => r.json());

                if (res.status === 'indexing_started') {
                    alert("Folder chunking & embedding started in the background.\nPlease refresh in a few moments to see the results.");
                    document.getElementById('index-dir-path').value = '';
                    fetchRagIndexCatalog();
                } else {
                    alert(`Failed: ${res.error}`);
                }
            } catch(e) {
                alert(`Error starting indexing: ${e}`);
            }
        }

        function handleRagSearch(e) {
            if (e.key === 'Enter') {
                searchRagIndex();
            }
        }

        async function searchRagIndex() {
            const query = document.getElementById('rag-search-query').value.trim();
            const select = document.getElementById('rag-search-model-select');
            const model = select.value;
            const projectDir = document.getElementById('rag-project-dir').value.trim();

            if (!query) return;

            const resultsContainer = document.getElementById('rag-search-results');
            resultsContainer.innerHTML = `
                <div style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                    🔎 Performing cosine similarity vector search...
                </div>
            `;

            try {
                const payload = { query: query, limit: 5 };
                if (model) payload.model = model;
                if (projectDir) payload.directory = projectDir;

                const matches = await fetch('/v1/db/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                }).then(r => r.json());

                if (matches.error) {
                    resultsContainer.innerHTML = `
                        <div style="text-align: center; color: var(--accent-rose); padding: 2rem;">
                            Error: ${matches.error}
                        </div>
                    `;
                    return;
                }

                if (matches.length === 0) {
                    resultsContainer.innerHTML = `
                        <div style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                            No matches found in the index for this embedding model.
                        </div>
                    `;
                    return;
                }

                resultsContainer.innerHTML = matches.map(m => {
                    return `
                        <div class="match-card">
                            <div class="match-header">
                                <span>Source: <strong>${m.file_path}</strong></span>
                                <span class="match-score">Similarity: ${m.similarity.toFixed(4)}</span>
                            </div>
                            <div class="match-text">${m.text}</div>
                        </div>
                    `;
                }).join('');

            } catch(e) {
                resultsContainer.innerHTML = `
                    <div style="text-align: center; color: var(--accent-rose); padding: 2rem;">
                        Connection error: ${e}
                    </div>
                `;
            }
        }

        window.addEventListener('DOMContentLoaded', init);
