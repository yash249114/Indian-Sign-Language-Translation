/**
 * Indian Sign Language Real-Time Client Application.
 * Supports:
 * 1. Fingerspelling Mode (Isolated Alphabets A-Z, Numbers 1-9)
 * 2. Word / Sign Mode (Multi-Hand Temporal CNN Sequence Recognition)
 * 3. Continuous ISL Mode (Boundary Segmentation + ISL Linguistic Grammar Layer)
 * 4. Gemini 3.5 Flash Lite Multilingual Translation & Local Offline TTS
 * 5. Interactive In-Browser Dataset Collection Tool
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements - Video & Camera
    const webcam = document.getElementById("webcam");
    const streamCanvas = document.getElementById("streamCanvas");
    const annotatedOverlay = document.getElementById("annotatedOverlay");
    const camPlaceholder = document.getElementById("camPlaceholder");
    const camToggleBtn = document.getElementById("camToggleBtn");
    const overlayToggle = document.getElementById("overlayToggle");
    const wsStatusBadge = document.getElementById("wsStatusBadge");
    const fpsTelemetry = document.getElementById("fpsTelemetry");

    // Mode Selector Elements
    const modeFingerspellingBtn = document.getElementById("modeFingerspellingBtn");
    const modeWordBtn = document.getElementById("modeWordBtn");
    const modeContinuousBtn = document.getElementById("modeContinuousBtn");
    const fingerspellingCard = document.getElementById("fingerspellingCard");
    const wordGlossCard = document.getElementById("wordGlossCard");
    const glyphLabel = document.getElementById("glyphLabel");
    const motionTelemetryBar = document.getElementById("motionTelemetryBar");
    const bufferFillTxt = document.getElementById("bufferFillTxt");
    const motionEnergyTxt = document.getElementById("motionEnergyTxt");
    const signTimingTxt = document.getElementById("signTimingTxt");

    // Detection Telemetry
    const signGlyph = document.getElementById("signGlyph");
    const confidenceVal = document.getElementById("confidenceVal");
    const confidenceBar = document.getElementById("confidenceBar");
    const stableSignTxt = document.getElementById("stableSignTxt");
    const stateTxt = document.getElementById("stateTxt");
    const top3List = document.getElementById("top3List");

    // Text & Gloss Buffers
    const fullTextDisplay = document.getElementById("fullTextDisplay");
    const currentToken = document.getElementById("currentToken");
    const glossTimeline = document.getElementById("glossTimeline");
    const interpretedEnglish = document.getElementById("interpretedEnglish");
    const grammarNotes = document.getElementById("grammarNotes");
    const uncertaintyWarning = document.getElementById("uncertaintyWarning");

    // Translation & Audio
    const translatedDisplay = document.getElementById("translatedDisplay");
    const targetLangSelect = document.getElementById("targetLangSelect");
    const translateBtn = document.getElementById("translateBtn");
    const transStatusBadge = document.getElementById("transStatusBadge");
    const cacheInfoTxt = document.getElementById("cacheInfoTxt");
    const transLatencyTxt = document.getElementById("transLatencyTxt");

    const modelLatencyVal = document.getElementById("modelLatencyVal");
    const mpLatencyVal = document.getElementById("mpLatencyVal");
    const e2eLatencyVal = document.getElementById("e2eLatencyVal");
    const pipelineFpsVal = document.getElementById("pipelineFpsVal");

    const spaceBtn = document.getElementById("spaceBtn");
    const backspaceBtn = document.getElementById("backspaceBtn");
    const clearBtn = document.getElementById("clearBtn");
    const undoGlossBtn = document.getElementById("undoGlossBtn");
    const clearGlossBtn = document.getElementById("clearGlossBtn");
    const speakBtn = document.getElementById("speakBtn");
    const replayBtn = document.getElementById("replayBtn");
    const ttsAudio = document.getElementById("ttsAudio");

    // Collector Modal Elements
    const openCollectorBtn = document.getElementById("openCollectorBtn");
    const collectorModal = document.getElementById("collectorModal");
    const closeModalBtn = document.getElementById("closeModalBtn");
    const cancelRecordBtn = document.getElementById("cancelRecordBtn");
    const startRecordBtn = document.getElementById("startRecordBtn");
    const recordGlossSelect = document.getElementById("recordGlossSelect");
    const signerIdInput = document.getElementById("signerIdInput");
    const lightingSelect = document.getElementById("lightingSelect");
    const handednessSelect = document.getElementById("handednessSelect");
    const recordStatusTxt = document.getElementById("recordStatusTxt");

    // State Variables
    let currentMode = "fingerspelling"; // "fingerspelling", "word", or "continuous"
    let isStreaming = false;
    let streamMedia = null;
    let ws = null;
    let frameSendInterval = null;
    const FRAME_RATE = 15; // FPS target for WebSocket streaming
    const canvasCtx = streamCanvas.getContext("2d");
    let lastAudioUrl = null;

    // Dataset Recording State
    let isCollectingClip = false;
    let collectedLandmarkFrames = [];

    // =========================================================================
    // Mode Switching Logic
    // =========================================================================
    function setMode(newMode) {
        currentMode = newMode;
        [modeFingerspellingBtn, modeWordBtn, modeContinuousBtn].forEach(b => b.classList.remove("active"));

        if (newMode === "fingerspelling") {
            modeFingerspellingBtn.classList.add("active");
            fingerspellingCard.style.display = "block";
            wordGlossCard.style.display = "none";
            motionTelemetryBar.style.display = "none";
            glyphLabel.textContent = "Current Character";
        } else if (newMode === "word") {
            modeWordBtn.classList.add("active");
            fingerspellingCard.style.display = "none";
            wordGlossCard.style.display = "block";
            motionTelemetryBar.style.display = "flex";
            glyphLabel.textContent = "Target Word Sign";
        } else if (newMode === "continuous") {
            modeContinuousBtn.classList.add("active");
            fingerspellingCard.style.display = "none";
            wordGlossCard.style.display = "block";
            motionTelemetryBar.style.display = "flex";
            glyphLabel.textContent = "Active Gloss Sign";
        }

        sendAction("set_mode", { target_mode: newMode });
    }

    modeFingerspellingBtn.addEventListener("click", () => setMode("fingerspelling"));
    modeWordBtn.addEventListener("click", () => setMode("word"));
    modeContinuousBtn.addEventListener("click", () => setMode("continuous"));

    // =========================================================================
    // WebSocket Connection Handling
    // =========================================================================
    function connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws/sign-stream`;

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            wsStatusBadge.textContent = "Live Stream Active";
            wsStatusBadge.className = "status-badge connected";
            sendAction("set_lang", { target_lang: targetLangSelect.value });
            sendAction("set_mode", { target_mode: currentMode });
        };

        ws.onclose = () => {
            wsStatusBadge.textContent = "Disconnected";
            wsStatusBadge.className = "status-badge disconnected";
            setTimeout(connectWebSocket, 2000);
        };

        ws.onerror = (err) => {
            console.error("[WS Error]", err);
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleServerMessage(data);
        };
    }

    function sendAction(actionType, extra = {}) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: actionType, mode: currentMode, ...extra }));
        }
    }

    // =========================================================================
    // Asynchronous Translation Requests
    // =========================================================================
    async function requestTranslation() {
        const targetLang = targetLangSelect.value;

        if (currentMode === "fingerspelling") {
            const text = fullTextDisplay.classList.contains("placeholder") ? "" : fullTextDisplay.textContent.trim();
            if (!text) {
                updateTranslationUI("", "ready", "● Ready", 0.0);
                return;
            }

            setTranslationStatus("translating", "● Translating via Gemini...");
            try {
                const resp = await fetch("/api/translate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: text, source_language: "en", target_language: targetLang })
                });
                const data = await resp.json();
                handleTranslationResponse(data);
            } catch (err) {
                updateTranslationUI("Translation unavailable", "error", "● Translation unavailable", 0.0);
            }
        } else {
            // Translate Gloss Sequence
            sendAction("translate");
        }
    }

    function handleTranslationResponse(data) {
        const translatedText = data.translated_text || "";
        const isCached = data.cached || false;
        const latency = data.latency_ms || 0.0;

        if (data.status === "success" || data.status === "offline_fallback" || data.status === "identity") {
            const statusLabel = isCached ? "● Cached translation" : "● Translation complete";
            const statusClass = isCached ? "cached" : "success";
            updateTranslationUI(translatedText, statusClass, statusLabel, latency);
        } else if (data.status === "cached") {
            updateTranslationUI(translatedText, "cached", "● Cached translation", latency);
        } else {
            updateTranslationUI(translatedText || "Translation unavailable", "error", "● Translation unavailable", latency);
        }

        fetchTelemetryStats();
    }

    function setTranslationStatus(cls, label) {
        if (transStatusBadge) {
            transStatusBadge.className = `trans-status ${cls}`;
            transStatusBadge.textContent = label;
        }
    }

    function updateTranslationUI(text, statusClass, statusLabel, latencyMs) {
        setTranslationStatus(statusClass, statusLabel);

        if (text && text.trim()) {
            translatedDisplay.textContent = text;
            translatedDisplay.classList.remove("placeholder");
        } else {
            translatedDisplay.textContent = "Translation output will appear here...";
            translatedDisplay.classList.add("placeholder");
        }

        if (transLatencyTxt) {
            transLatencyTxt.textContent = `${latencyMs} ms`;
        }
    }

    async function fetchTelemetryStats() {
        try {
            const resp = await fetch("/api/stats");
            const data = await resp.json();
            if (data.translation_telemetry && cacheInfoTxt) {
                const tel = data.translation_telemetry;
                cacheInfoTxt.textContent = `Cached: ${tel.cached_translations} | Calls Today: ${tel.translation_calls_today} / 500`;
            }
        } catch (e) {}
    }

    // =========================================================================
    // UI Update Handlers
    // =========================================================================
    function handleServerMessage(data) {
        if (data.type === "word_frame_result") {
            // Word & Continuous Temporal Result
            if (data.detected && (data.committed_gloss || data.state !== "IDLE")) {
                signGlyph.textContent = data.committed_gloss || data.top3?.[0]?.gloss || "-";
                const confPct = Math.round(data.confidence * 100);
                confidenceVal.textContent = `${confPct}%`;
                confidenceBar.style.width = `${confPct}%`;
            } else {
                signGlyph.textContent = "-";
                confidenceVal.textContent = "0.0%";
                confidenceBar.style.width = "0%";
            }

            stableSignTxt.textContent = data.committed_gloss || "-";
            if (stateTxt && data.state) {
                stateTxt.textContent = data.state;
                stateTxt.className = data.state === "COMMIT" ? "highlight-commit" :
                                    data.state === "COLLECTING" ? "highlight-stable" :
                                    data.state === "COOLDOWN" ? "highlight-locked" : "highlight-state";
            }

            // Motion telemetry
            if (bufferFillTxt) bufferFillTxt.textContent = `${data.buffer_fill}/30`;
            if (motionEnergyTxt) motionEnergyTxt.textContent = `${data.motion_energy}`;

            // Top candidates
            if (data.top3 && data.top3.length > 0) {
                top3List.innerHTML = data.top3.map(item => `
                    <div class="top3-item">
                        <span>${item.gloss}</span>
                        <span>${Math.round(item.confidence * 100)}%</span>
                    </div>
                `).join("");
            }

            // Update Gloss Timeline
            if (data.gloss_buffer) {
                updateGlossTimelineUI(data.gloss_buffer);
            }

            // Update Linguistic Interpretation
            if (data.interpretation) {
                updateGrammarUI(data.interpretation);
            }

            // Uncertainty warning
            if (uncertaintyWarning) {
                if (data.is_uncertain) {
                    uncertaintyWarning.style.display = "block";
                    uncertaintyWarning.textContent = "⚠️ Low confidence sign or uncatalogued motion. Verify signing boundary.";
                } else if (data.interpretation?.uncertainty_warning) {
                    uncertaintyWarning.style.display = "block";
                    uncertaintyWarning.textContent = `⚠️ ${data.interpretation.uncertainty_warning}`;
                } else {
                    uncertaintyWarning.style.display = "none";
                }
            }

            if (data.translated_text) {
                updateTranslationUI(data.translated_text, "success", "● Translation complete", 0.0);
            }

            // Telemetry breakdown
            updateTelemetryDisplay(data.telemetry);

            // Overlay
            if (data.annotated_frame_base64 && overlayToggle.checked) {
                annotatedOverlay.src = data.annotated_frame_base64;
                annotatedOverlay.style.display = "block";
            } else {
                annotatedOverlay.style.display = "none";
            }

        } else if (data.type === "frame_result") {
            // Fingerspelling Character Result
            if (data.detected && data.raw_sign) {
                signGlyph.textContent = data.raw_sign;
                const confPct = Math.round(data.confidence * 100);
                confidenceVal.textContent = `${confPct}%`;
                confidenceBar.style.width = `${confPct}%`;
            } else {
                signGlyph.textContent = "-";
                confidenceVal.textContent = "0.0%";
                confidenceBar.style.width = "0%";
            }

            stableSignTxt.textContent = data.stable_sign || "-";
            if (stateTxt && data.state) {
                stateTxt.textContent = data.state;
                stateTxt.className = data.state === "COMMIT_ONCE" ? "highlight-commit" :
                                    data.state === "LOCKED_HOLD" ? "highlight-locked" :
                                    data.state === "STABLE_SIGN" ? "highlight-stable" : "highlight-state";
            }

            if (data.top3 && data.top3.length > 0) {
                top3List.innerHTML = data.top3.map(item => `
                    <div class="top3-item">
                        <span>${item.sign}</span>
                        <span>${Math.round(item.confidence * 100)}%</span>
                    </div>
                `).join("");
            }

            updateBufferUI(data.buffer);
            if (data.translated_text) {
                updateTranslationUI(data.translated_text, "success", "● Translation complete", 0.0);
            }

            updateTelemetryDisplay(data.telemetry);

            if (data.annotated_frame_base64 && overlayToggle.checked) {
                annotatedOverlay.src = data.annotated_frame_base64;
                annotatedOverlay.style.display = "block";
            } else {
                annotatedOverlay.style.display = "none";
            }

        } else if (data.type === "gloss_translation_result") {
            const tr = data.translation;
            if (tr) {
                handleTranslationResponse(tr);
            }
            if (data.gloss_buffer) {
                updateGlossTimelineUI(data.gloss_buffer);
            }
        } else if (data.type === "buffer_update") {
            if (data.buffer) updateBufferUI(data.buffer);
            if (data.gloss_buffer) updateGlossTimelineUI(data.gloss_buffer);
        }
    }

    function updateTelemetryDisplay(tel) {
        if (!tel) return;
        if (modelLatencyVal) modelLatencyVal.textContent = `${tel.model_inference_ms} ms`;
        if (mpLatencyVal) mpLatencyVal.textContent = `${tel.mediapipe_tracking_ms} ms`;
        if (e2eLatencyVal) e2eLatencyVal.textContent = `${tel.e2e_latency_ms} ms`;
        if (pipelineFpsVal) pipelineFpsVal.textContent = `${tel.stream_fps} FPS`;
        if (fpsTelemetry) fpsTelemetry.textContent = `${tel.stream_fps} FPS`;
    }

    function updateBufferUI(buffer) {
        if (buffer) {
            if (buffer.full_text && buffer.full_text.trim()) {
                fullTextDisplay.textContent = buffer.full_text;
                fullTextDisplay.classList.remove("placeholder");
            } else {
                fullTextDisplay.textContent = "Recognized characters will appear here...";
                fullTextDisplay.classList.add("placeholder");
            }
            currentToken.textContent = buffer.current_word || "-";
        }
    }

    function updateGlossTimelineUI(glossBuffer) {
        if (!glossBuffer || !glossBuffer.tokens || glossBuffer.tokens.length === 0) {
            glossTimeline.innerHTML = `<span class="placeholder" style="color: var(--text-muted); font-size: 0.85rem;">Perform an ISL sign to see recognized glosses...</span>`;
            interpretedEnglish.textContent = "-";
            grammarNotes.textContent = "Signs will be parsed according to ISL grammar rules.";
            return;
        }

        glossTimeline.innerHTML = glossBuffer.tokens.map((t, idx) => {
            const timeStr = new Date(t.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            return `
                <div class="gloss-chip">
                    <span>${idx + 1}. ${t.gloss}</span>
                    <span class="gloss-chip-time">${timeStr}</span>
                </div>
            `;
        }).join("");
    }

    function updateGrammarUI(interp) {
        if (!interp) return;
        if (interpretedEnglish) {
            interpretedEnglish.textContent = interp.interpreted_english || "-";
        }
        if (grammarNotes) {
            grammarNotes.textContent = `Pattern: [${interp.syntactic_structure}] • ${interp.grammatical_notes}`;
        }
    }

    // =========================================================================
    // Webcam Life-Cycle & Stream Loop
    // =========================================================================
    async function startWebcam() {
        try {
            streamMedia = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
                audio: false
            });
            webcam.srcObject = streamMedia;
            camPlaceholder.style.display = "none";
            camToggleBtn.textContent = "Stop Camera";
            camToggleBtn.className = "btn btn-danger btn-sm";
            isStreaming = true;

            webcam.onloadedmetadata = () => {
                streamCanvas.width = webcam.videoWidth || 640;
                streamCanvas.height = webcam.videoHeight || 480;
                startFrameLoop();
            };
        } catch (err) {
            alert("Could not access webcam: " + err.message);
        }
    }

    function stopWebcam() {
        if (streamMedia) {
            streamMedia.getTracks().forEach(track => track.stop());
            streamMedia = null;
        }
        if (frameSendInterval) {
            clearInterval(frameSendInterval);
            frameSendInterval = null;
        }
        webcam.srcObject = null;
        annotatedOverlay.style.display = "none";
        camPlaceholder.style.display = "flex";
        camToggleBtn.textContent = "Start Camera";
        camToggleBtn.className = "btn btn-primary btn-sm";
        isStreaming = false;
        signGlyph.textContent = "-";
        confidenceVal.textContent = "0.0%";
        confidenceBar.style.width = "0%";
        if (stateTxt) stateTxt.textContent = "NO_SIGN";
    }

    function startFrameLoop() {
        if (frameSendInterval) clearInterval(frameSendInterval);

        frameSendInterval = setInterval(() => {
            if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) return;

            canvasCtx.drawImage(webcam, 0, 0, streamCanvas.width, streamCanvas.height);
            const imageBase64 = streamCanvas.toDataURL("image/jpeg", 0.7);

            ws.send(JSON.stringify({
                image: imageBase64,
                mode: currentMode,
                include_annotated: overlayToggle.checked,
                target_lang: targetLangSelect.value
            }));
        }, 1000 / FRAME_RATE);
    }

    // =========================================================================
    // Action Event Listeners
    // =========================================================================
    camToggleBtn.addEventListener("click", () => {
        if (isStreaming) stopWebcam();
        else startWebcam();
    });

    spaceBtn.addEventListener("click", () => {
        sendAction("space");
        setTimeout(requestTranslation, 100);
    });

    backspaceBtn.addEventListener("click", () => {
        sendAction("backspace");
    });

    clearBtn.addEventListener("click", () => {
        sendAction("clear");
        updateTranslationUI("", "ready", "● Ready", 0.0);
    });

    undoGlossBtn.addEventListener("click", () => {
        sendAction("backspace");
        setTimeout(requestTranslation, 100);
    });

    clearGlossBtn.addEventListener("click", () => {
        sendAction("clear");
        updateTranslationUI("", "ready", "● Ready", 0.0);
    });

    translateBtn.addEventListener("click", () => {
        requestTranslation();
    });

    targetLangSelect.addEventListener("change", () => {
        sendAction("set_lang", { target_lang: targetLangSelect.value });
        requestTranslation();
    });

    // Offline Text-to-Speech & Replay
    speakBtn.addEventListener("click", async () => {
        let textToSpeak = translatedDisplay.classList.contains("placeholder") ? "" : translatedDisplay.textContent.trim();
        if (!textToSpeak) {
            textToSpeak = currentMode === "fingerspelling" ?
                (fullTextDisplay.classList.contains("placeholder") ? "" : fullTextDisplay.textContent.trim()) :
                (interpretedEnglish.textContent !== "-" ? interpretedEnglish.textContent : "");
        }
        if (!textToSpeak) return;

        speakBtn.disabled = true;
        speakBtn.innerHTML = `<span class="btn-icon">⏳</span> Synthesizing...`;

        try {
            const resp = await fetch("/api/tts", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: textToSpeak, language: "en" })
            });
            const data = await resp.json();

            if (data.audio_file) {
                lastAudioUrl = data.audio_file + `?t=${Date.now()}`;
                ttsAudio.src = lastAudioUrl;
                ttsAudio.play().catch(() => speakBrowserFallback(textToSpeak));
            } else {
                speakBrowserFallback(textToSpeak);
            }
        } catch (err) {
            speakBrowserFallback(textToSpeak);
        } finally {
            speakBtn.disabled = false;
            speakBtn.innerHTML = `<span class="btn-icon">🔊</span> Synthesize Speech (Local Offline TTS)`;
        }
    });

    replayBtn.addEventListener("click", () => {
        if (lastAudioUrl) {
            ttsAudio.currentTime = 0;
            ttsAudio.play().catch(() => {});
        } else {
            speakBtn.click();
        }
    });

    function speakBrowserFallback(text) {
        if ("speechSynthesis" in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(utterance);
        }
    }

    // =========================================================================
    // Dataset Collection Modal Logic
    // =========================================================================
    openCollectorBtn.addEventListener("click", () => {
        collectorModal.style.display = "flex";
    });

    closeModalBtn.addEventListener("click", () => {
        collectorModal.style.display = "none";
    });

    cancelRecordBtn.addEventListener("click", () => {
        collectorModal.style.display = "none";
    });

    startRecordBtn.addEventListener("click", async () => {
        if (!isStreaming) {
            alert("Please start the webcam first before capturing signs!");
            return;
        }

        const gloss = recordGlossSelect.value;
        const signerId = signerIdInput.value.trim() || "signer_01";
        const lighting = lightingSelect.value;
        const handedness = handednessSelect.value;

        startRecordBtn.disabled = true;
        recordStatusTxt.textContent = `Recording 30 frames for ${gloss}... Perform the sign now!`;
        recordStatusTxt.style.color = "#dc2626";

        // Collect 30 consecutive feature frames from the stream
        setTimeout(async () => {
            try {
                const resp = await fetch("/api/dataset/record-clip", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        gloss: gloss,
                        signer_id: signerId,
                        lighting: lighting,
                        handedness: handedness,
                        notes: "Captured via in-browser dataset collection tool"
                    })
                });
                const data = await resp.json();
                recordStatusTxt.textContent = `Successfully saved ${gloss} clip into manifest! (Assigned split: ${data.clip?.split})`;
                recordStatusTxt.style.color = "#059669";
            } catch (err) {
                recordStatusTxt.textContent = `Error recording clip: ${err.message}`;
                recordStatusTxt.style.color = "#dc2626";
            } finally {
                startRecordBtn.disabled = false;
            }
        }, 1500);
    });

    // Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
        if (e.code === "Space") {
            e.preventDefault();
            spaceBtn.click();
        } else if (e.code === "Backspace") {
            e.preventDefault();
            if (currentMode === "fingerspelling") backspaceBtn.click();
            else undoGlossBtn.click();
        } else if (e.code === "Escape") {
            e.preventDefault();
            clearBtn.click();
        }
    });

    // Initialize WebSocket connection and fetch stats on start
    connectWebSocket();
    fetchTelemetryStats();
});
