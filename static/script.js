document.addEventListener('DOMContentLoaded', function() {
    // Common variables
    let selectedModel = '';
    let selectedEvalType = 'natural_chat';
    
    // Fetch available models if on Evaluation page or Direct Chat page
    const modelSelect = document.getElementById('model-select');
    const evalTypeSelect = document.getElementById('eval-type-select');
    const clearChatBtn = document.getElementById('clear-chat');
    
    // Page detection
    const isEvaluationPage = document.querySelector('.comparison-container') !== null || document.querySelector('.comparison-interface') !== null;
    const isDirectChatPage = document.querySelector('h1')?.textContent.includes('Direct Chat') || false;
    
    console.log("Page detection:", { isEvaluationPage, isDirectChatPage });
    console.log("Model select element:", modelSelect);
    
    if (modelSelect) {
        console.log("Fetching available models...");
        fetchAvailableModels();
    } else {
        console.warn("Model select element not found in DOM");
    }
    
    if (evalTypeSelect) {
        evalTypeSelect.addEventListener('change', function() {
            selectedEvalType = this.value;
            updateAnalysis(); // Update analysis when eval type changes
        });
    }
    
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', function() {
            const chatHistory = document.getElementById('chat-history');
            if (chatHistory) {
                chatHistory.innerHTML = '';
                // Add a welcome message
                const welcomeMsg = document.createElement('div');
                welcomeMsg.classList.add('message', 'ai-message');
                welcomeMsg.textContent = 'Hello! I\'m your AI clone. How can I help you today?';
                chatHistory.appendChild(welcomeMsg);
            }
        });
    }
    
    async function fetchAvailableModels() {
        console.log("Starting fetchAvailableModels...");
        
        if (!modelSelect) {
            console.error("Model select element not found");
            return;
        }
        
        try {
            console.log("Sending request to /api/models...");
            const response = await fetch('/api/models');
            console.log("Response status:", response.status);
            
            if (!response.ok) {
                throw new Error(`Failed to fetch models: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log("Models data received:", data);
            
            // Clear existing options
            modelSelect.innerHTML = '';
            
            if (!data.models || data.models.length === 0) {
                console.log("No models available");
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'No models available. Please create one first.';
                option.style.color = '#999';
                modelSelect.appendChild(option);
                modelSelect.disabled = true;
            } else {
                console.log("Adding models to select:", data.models);
                
                // Add default option
                const defaultOption = document.createElement('option');
                defaultOption.value = '';
                defaultOption.textContent = 'Select an AI Clone...';
                defaultOption.disabled = true;
                defaultOption.selected = true;
                modelSelect.appendChild(defaultOption);
                
                // Add model options
                data.models.forEach((model, index) => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                    console.log(`Added model option: ${model}`);
                });
                
                modelSelect.disabled = false;
                
                // Auto-select first model if available
                if (data.models.length > 0) {
                    selectedModel = data.models[0];
                    modelSelect.value = selectedModel;
                    console.log("Auto-selected model:", selectedModel);
                }
            }
        } catch (error) {
            console.error('Error fetching models:', error);
            modelSelect.innerHTML = '';
            const option = document.createElement('option');
            option.value = '';
            option.textContent = `Error loading models: ${error.message}`;
            option.style.color = '#f00';
            modelSelect.appendChild(option);
            modelSelect.disabled = true;
        }
    }
    
    // Model selection change handler
    if (modelSelect) {
        modelSelect.addEventListener('change', function() {
            selectedModel = this.value;
            console.log("Model selected:", selectedModel);
            
            // Update status indicator
            const statusText = document.getElementById('status-text');
            if (statusText) {
                if (selectedModel) {
                    statusText.textContent = `Ready for evaluation with ${selectedModel}`;
                } else {
                    statusText.textContent = 'Please select a model';
                }
            }
            
            // Update analysis when model changes
            if (selectedModel) {
                updateAnalysis();
            }
        });
    }
    
    // Setup page logic for upload/select/chat
    if (document.getElementById('upload-form')) {
        const uploadForm = document.getElementById('upload-form');
        const participantSelect = document.getElementById('participant-select');
        const participantsDiv = document.getElementById('participants');
        const participantForm = document.getElementById('participant-form');
        const chatSection = document.getElementById('chat-section');
        const chatBox = document.getElementById('chat-box');
        const chatForm = document.getElementById('chat-form');
        const userMessage = document.getElementById('user-message');
        const fileInput = document.getElementById('file-input');
        const uploadArea = document.querySelector('.upload-area');

        let aiParticipant = null;
        let modelName = null;

        // File upload area click handler
        if (uploadArea && fileInput) {
            uploadArea.addEventListener('click', function(e) {
                if (e.target !== fileInput) {
                    fileInput.click();
                }
            });

            // Drag and drop handlers
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                uploadArea.classList.add('drag-over');
            });

            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('drag-over');
            });

            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('drag-over');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    updateFileDisplay(files[0]);
                }
            });

            // File input change handler
            fileInput.addEventListener('change', function(e) {
                if (e.target.files.length > 0) {
                    updateFileDisplay(e.target.files[0]);
                }
            });
        }

        // Function to update file display
        function updateFileDisplay(file) {
            const uploadContent = document.querySelector('.upload-content');
            if (uploadContent && file) {
                uploadContent.innerHTML = `
                    <div class="file-selected">
                        <svg class="file-icon" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M4 4a2 2 0 012-2h8l4 4v10a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm8 0v4h4l-4-4z" clip-rule="evenodd" />
                        </svg>
                        <div class="file-info">
                            <h4>${file.name}</h4>
                            <p>${formatFileSize(file.size)}</p>
                        </div>
                        <svg class="file-check" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                        </svg>
                    </div>
                `;
            }
        }

        // Helper function to format file size
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        uploadForm.onsubmit = async function(e) {
            e.preventDefault();
            
            const submitButton = uploadForm.querySelector('button[type="submit"]');
            const originalText = submitButton.innerHTML;
            
            // Check if file is selected
            if (!fileInput.files || fileInput.files.length === 0) {
                alert('Please select a file first');
                return;
            }

            try {
                // Show loading state
                submitButton.innerHTML = `
                    <div class="button-spinner"></div>
                    <span>Analyzing...</span>
                `;
                submitButton.disabled = true;

                const formData = new FormData(uploadForm);
                const res = await fetch('/upload', { 
                    method: 'POST', 
                    body: formData 
                });
                
                if (!res.ok) {
                    throw new Error(`Upload failed: ${res.status} ${res.statusText}`);
                }
                
                const data = await res.json();
                
                if (data.participants) {
                    // Hide upload step and show participant selection
                    document.getElementById('step-upload').style.display = 'none';
                    document.getElementById('step-participant').style.display = 'block';
                    
                    // Update progress steps
                    document.querySelector('.step-item[data-step="1"]').classList.add('completed');
                    document.querySelector('.step-item[data-step="2"]').classList.add('active');
                    
                    // Populate participants
                    participantsDiv.innerHTML = '';
                    data.participants.forEach((p, idx) => {
                        const participantName = typeof p === 'string' ? p : (p.name || JSON.stringify(p));
                        const participantCard = document.createElement('div');
                        participantCard.className = 'participant-card';
                        participantCard.innerHTML = `
                            <input type="radio" name="participant" value="${participantName}" id="participant-${idx}" ${idx===0?'checked':''}>
                            <label for="participant-${idx}" class="participant-label">
                                <div class="participant-avatar">
                                    <svg viewBox="0 0 20 20" fill="currentColor">
                                        <path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd" />
                                    </svg>
                                </div>
                                <div class="participant-name">${participantName}</div>
                            </label>
                        `;
                        participantsDiv.appendChild(participantCard);
                    });
                } else {
                    throw new Error('No participants found in the uploaded file');
                }
            } catch (error) {
                console.error('Upload error:', error);
                alert(`Error uploading file: ${error.message}`);
            } finally {
                // Restore button state
                submitButton.innerHTML = originalText;
                submitButton.disabled = false;
            }
        };
          // Get all needed elements for the processing section
        const processingSection = document.getElementById('processing-section');
        const progressBar = document.getElementById('progress-bar');
        const statusMessage = document.getElementById('status-message');
        const currentStep = document.getElementById('current-step');
        const totalSteps = document.getElementById('total-steps');
        const chunksProcessed = document.getElementById('chunks-processed');
        const totalChunks = document.getElementById('total-chunks');
        const batchesProcessed = document.getElementById('batches-processed');
        const totalBatches = document.getElementById('total-batches');
        const continueToChat = document.getElementById('continue-to-chat');
        
        let processingInterval;
        
        // Chunk size slider functionality
        const chunkSizeSlider = document.getElementById('chunk-size-slider');
        const chunkSizeValue = document.getElementById('chunk-size-value');
        
        // Update the displayed value when the slider changes
        if (chunkSizeSlider) {
            // Calculate token size based on slider value
            function updateChunkSizeDescription(value) {
                const minSize = 800;
                const maxSize = 3500;
                const tokenSize = Math.round(minSize + ((maxSize - minSize) * (value - 1) / 9));
                
                // Update the value display with the calculated token size
                chunkSizeValue.textContent = value;
                
                // Add a tooltip or additional description
                chunkSizeValue.title = `Approximately ${tokenSize} tokens per chunk`;
                
                // Add visual feedback based on the selected size
                if (value <= 3) {
                    chunkSizeValue.style.backgroundColor = '#c8e6c9'; // Light green for small chunks
                } else if (value <= 7) {
                    chunkSizeValue.style.backgroundColor = '#e8f0fe'; // Default blue for medium chunks
                } else {
                    chunkSizeValue.style.backgroundColor = '#ffecb3'; // Light amber for large chunks
                }
            }
            
            // Initialize with starting value
            updateChunkSizeDescription(chunkSizeSlider.value);
            
            // Update when slider changes
            chunkSizeSlider.oninput = function() {
                updateChunkSizeDescription(this.value);
            };
        }
        
        participantForm.onsubmit = async function(e) {
            e.preventDefault();
            aiParticipant = participantForm.participant.value;
            modelName = participantForm.model_name.value;
            const chunkSize = participantForm.chunk_size.value;
            
            try {
                // Step 1: Initialize basic model structure
                statusMessage.textContent = "Initializing your AI clone...";
                const initRes = await fetch('/select_participant', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        participant: aiParticipant, 
                        model_name: modelName,
                        chunk_size: parseInt(chunkSize)
                    })
                });
                
                const initData = await initRes.json();
                if (initData.status === 'processing' || initData.status === 'ready') {
                    // Hide participant selection and show processing section
                    participantSelect.style.display = 'none';
                    processingSection.style.display = '';
                      // Step 2: Start background processing
                    const processRes = await fetch('/api/process_model', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            model_name: modelName, 
                            participant: aiParticipant,
                            chunk_size: parseInt(chunkSizeSlider.value)
                        })
                    });
                    
                    const processData = await processRes.json();
                    if (processData.status === 'processing') {
                        // Start polling for updates
                        statusMessage.textContent = "Processing your AI clone...";
                        startProcessingUpdates(modelName);
                    } else {
                        statusMessage.textContent = `Error starting processing: ${processData.message || 'Unknown error'}`;
                    }
                } else {
                    statusMessage.textContent = `Error: ${initData.msg || 'Unknown error during initialization'}`;
                }
            } catch (error) {
                console.error('Error during setup:', error);
                statusMessage.textContent = `Error: ${error.message || 'Unknown error during setup'}`;
            }
        };
          // Function to poll for processing updates
        function startProcessingUpdates(model) {
            clearInterval(processingInterval);
            
            // Setup polling with increasing frequency based on stage
            let pollFrequency = 1000; // Start with 1 second during chunking
            
            processingInterval = setInterval(async () => {
                try {
                    const response = await fetch(`/api/process_status?model_name=${model}`);
                    if (!response.ok) {
                        throw new Error(`Failed to get status: ${response.status} ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    updateProgressDisplay(data);
                    
                    // Adjust polling frequency based on the stage
                    // Chunking/preparation stages need more frequent updates
                    // Summarization can be less frequent as it might take longer
                    if (data.progress && data.progress.current_step === 'chunking') {
                        pollFrequency = 500; // More frequent during chunking
                    } else if (data.progress && data.progress.current_step === 'summarizing') {
                        pollFrequency = 1000; // Every second during summarization
                    } else {
                        pollFrequency = 2000; // Otherwise every 2 seconds
                    }
                    
                    // Check if processing is complete
                    if (data.status === 'completed') {
                        clearInterval(processingInterval);
                        continueToChat.style.display = 'block';
                    } else if (data.status === 'error') {
                        clearInterval(processingInterval);
                        statusMessage.textContent = `Error: ${data.error || 'Unknown error during processing'}`;
                    }
                } catch (error) {
                    console.error('Error checking processing status:', error);
                }
            }, pollFrequency);
        }
          // Function to update the progress display
        function updateProgressDisplay(data) {
            if (!data || !data.progress) return;
            
            const progress = data.progress;
            const progressPercent = (progress.step / progress.total_steps) * 100;
            
            // Update progress bar
            progressBar.style.width = `${progressPercent}%`;
            
            // Update status message
            statusMessage.textContent = progress.message || "Processing...";
            
            // Extract token count and chunk size from message if available
            if (progress.message && progress.message.includes("Total tokens:")) {
                // Extract token and chunk size information to display separately
                try {
                    const tokenMatch = progress.message.match(/Total tokens: (\d+)/);
                    const chunkSizeMatch = progress.message.match(/Chunk size: (\d+)/);
                    
                    if (tokenMatch && tokenMatch[1]) {
                        // Create or update token count element
                        let tokenElement = document.getElementById('token-count-info');
                        if (!tokenElement) {
                            tokenElement = document.createElement('p');
                            tokenElement.id = 'token-count-info';
                            document.getElementById('step-details').appendChild(tokenElement);
                        }
                        tokenElement.innerHTML = `<strong>Total tokens:</strong> ${parseInt(tokenMatch[1]).toLocaleString()}`;
                    }
                    
                    if (chunkSizeMatch && chunkSizeMatch[1]) {
                        // Create or update chunk size element
                        let chunkSizeElement = document.getElementById('chunk-size-info');
                        if (!chunkSizeElement) {
                            chunkSizeElement = document.createElement('p');
                            chunkSizeElement.id = 'chunk-size-info';
                            document.getElementById('step-details').appendChild(chunkSizeElement);
                        }
                        chunkSizeElement.innerHTML = `<strong>Chunk size:</strong> ${parseInt(chunkSizeMatch[1]).toLocaleString()} tokens`;
                    }
                } catch (e) {
                    console.error("Error parsing token information:", e);
                }
            }
            
            // Update step information
            currentStep.textContent = progress.current_step || progress.step || "-";
            totalSteps.textContent = progress.total_steps || "4";
            
            // Update chunk information
            chunksProcessed.textContent = progress.chunks_processed || "0";
            totalChunks.textContent = progress.total_chunks || "0";
            
            // Update batch information
            batchesProcessed.textContent = progress.batches_processed || "0";
            totalBatches.textContent = progress.total_batches || "0";
        }
        
        // Continue to chat button handler
        continueToChat.addEventListener('click', function() {
            processingSection.style.display = 'none';
            chatSection.style.display = '';
        });chatForm.onsubmit = async function(e) {
            e.preventDefault();
            const msg = userMessage.value;
            chatBox.innerHTML += `<div><b>You:</b> ${msg}</div>`;
            userMessage.value = '';
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg, model: modelName })
            });
            const data = await res.json();
            chatBox.innerHTML += `<div><b>${modelName}:</b> ${data.langchain}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
            
            // After creating a model, add it to the models list
            updateModelsList(modelName);
        };
        
        // Function to update models list after creating a new model
        async function updateModelsList(newModel) {
            try {
                // Fetch current models
                const response = await fetch('/api/models');
                if (!response.ok) throw new Error('Failed to fetch models');
                const data = await response.json();
                
                // Check if model already exists
                if (!data.models.includes(newModel)) {
                    // Add new model to the list
                    data.models.push(newModel);
                    
                    // Update models.json
                    await fetch('/api/update_models', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ models: data.models })
                    });
                }
            } catch (error) {
                console.error('Error updating models list:', error);
            }
        }
    }
    // Main chat functionality
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatHistory = document.getElementById('chat-history');
    
    // Check if elements exist and log them for debugging
    console.log("Critical elements:", { 
        userInput: userInput !== null, 
        sendBtn: sendBtn !== null, 
        chatHistory: chatHistory !== null 
    });
    
    // Response elements for evaluation
    const responseA = document.getElementById('response-a');
    const responseB = document.getElementById('response-b');
    const baselineResponse = document.getElementById('baseline-response');
    const langchainResponse = document.getElementById('langchain-response');
    const comparisonContainer = document.getElementById('comparison-container');
    const voteButtons = document.querySelectorAll('.vote-btn');
    
    let currentMessage = '';
    let currentResponses = {
        baseline: '',
        langchain: ''
    };
    
    // Initialize the send button functionality based on the page type
    function initializeSendButton() {
        console.log("Initializing send button...");
        
        if (!sendBtn) {
            console.error("Send button not found in the DOM");
            return;
        }
        
        if (!userInput) {
            console.error("User input not found in the DOM");
            return;
        }
        
        // Clear any existing listeners by cloning the button
        const oldButton = sendBtn;
        const newButton = oldButton.cloneNode(true);
        oldButton.parentNode.replaceChild(newButton, oldButton);
        
        // Get fresh reference
        const freshSendBtn = document.getElementById('send-btn');
        
        // Add the appropriate event listener based on page type
        if (isEvaluationPage) {
            console.log("Setting up send button for evaluation page");
            
            // Using both onclick and addEventListener for redundancy
            freshSendBtn.onclick = function() {
                handleSendButtonClick('evaluation');
            };
            
            freshSendBtn.addEventListener('click', function() {
                handleSendButtonClick('evaluation');
            });
        } else {
            console.log("Setting up send button for direct chat page");
            
            // Using both onclick and addEventListener for redundancy
            freshSendBtn.onclick = function() {
                handleSendButtonClick('direct');
            };
            
            freshSendBtn.addEventListener('click', function() {
                handleSendButtonClick('direct');
            });
        }
        
        // Set up Enter key handling
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                handleSendButtonClick(isEvaluationPage ? 'evaluation' : 'direct');
            }
        });
        
        console.log("Send button initialization complete");
    }
    
    // Unified function to handle send button clicks
    function handleSendButtonClick(mode) {
        if (!userInput) {
            console.error("User input element not found");
            return;
        }
        
        const message = userInput.value.trim();
        console.log(`Send button clicked in ${mode} mode with message: ${message}`);
        
        if (!message) {
            console.log("Empty message, not sending");
            return;
        }
        
        if (!selectedModel && modelSelect) {
            selectedModel = modelSelect.value;
        }
        
        if (!selectedModel) {
            console.error("No model selected");
            addMessage("Please select a model first", "ai");
            return;
        }
        
        if (mode === 'evaluation') {
            console.log("Sending evaluation message");
            sendEvaluationMessage(message);
        } else {
            console.log("Sending direct message");
            sendDirectMessage(message);
        }
        
        userInput.value = '';
    }
    
    // Function to show typing indicator
    function showTypingIndicator() {
        const chatHistory = document.getElementById('chat-history');
        if (!chatHistory) return;
        
        const typingDiv = document.createElement('div');
        typingDiv.classList.add('typing-indicator');
        typingDiv.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        chatHistory.appendChild(typingDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return typingDiv;
    }
    
    // Function to remove typing indicator
    function removeTypingIndicator(indicator) {
        if (indicator && indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
        }
    }
    
    // Function to add a message to the chat history
    function addMessage(content, type) {
        const chatHistory = document.getElementById('chat-history');
        if (!chatHistory) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${type}-message`);
        messageDiv.textContent = content;
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
    
    // Function to send message for evaluation (shows both responses)
    async function sendEvaluationMessage(message) {
        console.log("Starting evaluation message send...");
        
        if (!selectedModel) {
            addMessage('Please select a model first.', 'ai');
            return;
        }
        
        // Store current message for evaluation
        currentMessage = message;
        
        // Show user message
        addMessage(message, 'user');
        
        // Show typing indicator
        const typingIndicator = showTypingIndicator();
        
        // Update the question display in comparison interface
        const currentQuestionElement = document.getElementById('current-question');
        if (currentQuestionElement) {
            currentQuestionElement.textContent = message;
        }
        
        try {
            console.log(`Fetching responses for model: ${selectedModel}`);
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message,
                    model: selectedModel 
                })
            });
            
            if (!response.ok) {
                throw new Error(`Failed to get response: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Received API response:", data);
            
            // Remove typing indicator
            removeTypingIndicator(typingIndicator);
            
            if (data.baseline && data.langchain) {
                // Store responses for evaluation
                currentResponses.baseline = data.baseline;
                currentResponses.langchain = data.langchain;
                
                // Update the comparison container with the responses
                const baselineResponseElement = document.getElementById('baseline-response');
                const langchainResponseElement = document.getElementById('langchain-response');
                const comparisonContainer = document.getElementById('comparison-container');
                
                if (baselineResponseElement && langchainResponseElement && comparisonContainer) {
                    // Clear loading states and show actual responses
                    baselineResponseElement.innerHTML = `<p>${data.baseline}</p>`;
                    langchainResponseElement.innerHTML = `<p>${data.langchain}</p>`;
                    
                    // Enable vote buttons
                    const voteButtons = document.querySelectorAll('.vote-button');
                    voteButtons.forEach(btn => {
                        btn.disabled = false;
                    });
                    
                    // Update progress steps
                    const progressSteps = document.querySelectorAll('.progress-step');
                    progressSteps.forEach((step, index) => {
                        if (index <= 2) { // Mark first 3 steps as active
                            step.classList.add('active');
                        }
                    });
                    
                    // Show comparison container
                    comparisonContainer.style.display = 'block';
                    
                    // Scroll to comparison container smoothly
                    setTimeout(() => {
                        comparisonContainer.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }, 300);
                    
                    // Add a message in chat indicating comparison is ready
                    addMessage('✅ Responses generated! Please compare the responses below and choose which one is better.', 'ai');
                    
                } else {
                    console.error("Comparison elements not found", {
                        baselineResponse: baselineResponseElement !== null,
                        langchainResponse: langchainResponseElement !== null,
                        comparisonContainer: comparisonContainer !== null
                    });
                    addMessage('Error: Could not display comparison interface. Please try again.', 'ai');
                }
            } else {
                console.error("Missing baseline or langchain response in API response:", data);
                addMessage('Error: Incomplete responses from the models. Please try again.', 'ai');
            }
        } catch (error) {
            console.error('Error during evaluation:', error);
            removeTypingIndicator(typingIndicator);
            addMessage(`Sorry, there was an error: ${error.message}`, 'ai');
        }
    }
    
    // Function to send message in direct chat mode
    async function sendDirectMessage(message) {
        console.log("Starting direct message send...");
        
        if (!selectedModel) {
            addMessage('Please select a model first.', 'ai');
            return;
        }
        
        // Show user message
        addMessage(message, 'user');
        
        // Show typing indicator
        const typingIndicator = showTypingIndicator();
        
        try {
            console.log(`Sending direct message to model: ${selectedModel}`);
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message,
                    model: selectedModel 
                })
            });
            
            if (!response.ok) {
                throw new Error(`Failed to get response: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Received API response:", data);
            
            // Remove typing indicator
            removeTypingIndicator(typingIndicator);
            
            // In direct chat, we use the langchain (cloned) response
            if (data.langchain) {
                addMessage(data.langchain, 'ai');
            } else {
                console.error("Missing langchain response");
                addMessage('Error: Missing model response', 'ai');
            }
        } catch (error) {
            console.error('Error during direct chat:', error);
            removeTypingIndicator(typingIndicator);
            addMessage(`Sorry, there was an error: ${error.message}`, 'ai');
        }
    }
    
    // Initialize send button when the DOM is fully loaded
    initializeSendButton();
    
    // Also initialize after a short delay to ensure everything is loaded
    setTimeout(initializeSendButton, 500);
    
    // Event listeners for vote buttons - Updated to use correct selectors
    const voteButtonsQuery = document.querySelectorAll('.vote-button');
    voteButtonsQuery.forEach(button => {
        button.addEventListener('click', async function() {
            const winner = this.getAttribute('data-model');
            console.log(`Vote submitted for: ${winner}`);
            
            try {
                // Disable both buttons to prevent double submission
                voteButtonsQuery.forEach(btn => {
                    btn.disabled = true;
                    btn.style.opacity = '0.7';
                });
                
                // Add visual feedback
                this.style.backgroundColor = winner === 'baseline' ? '#f59e0b' : '#10b981';
                const buttonText = this.querySelector('span');
                if (buttonText) {
                    buttonText.textContent = 'Selected ✓';
                }
                
                // Submit evaluation
                const response = await fetch('/api/evaluate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ 
                        winner,
                        message: currentMessage,
                        baseline_response: currentResponses.baseline,
                        langchain_response: currentResponses.langchain,
                        model_name: selectedModel,
                        eval_type: selectedEvalType
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to submit evaluation');
                }
                
                // Update the final progress step
                const finalStep = document.querySelector('.progress-step:last-child');
                if (finalStep) {
                    finalStep.classList.add('active');
                }
                
                // Update analysis after successful evaluation
                updateAnalysis();
                
                // Hide comparison container with a smooth transition
                const comparisonContainer = document.getElementById('comparison-container');
                if (comparisonContainer) {
                    comparisonContainer.style.opacity = '0';
                    setTimeout(() => {
                        comparisonContainer.style.display = 'none';
                        comparisonContainer.style.opacity = '1';
                        
                        // Reset for next evaluation
                        resetComparisonInterface();
                    }, 500);
                }
                
                // Show the chosen response in chat
                const chosenResponse = winner === 'baseline' ? currentResponses.baseline : currentResponses.langchain;
                const modelLabel = winner === 'baseline' ? 'Baseline Model' : 'AI Clone';
                addMessage(`${modelLabel}: ${chosenResponse}`, 'ai');
                
                // Show a brief success message
                const successMessage = document.createElement('div');
                successMessage.className = 'comparison-result ' + winner;
                successMessage.innerHTML = `
                    <strong>Evaluation Complete!</strong><br>
                    You selected the ${winner === 'baseline' ? 'Baseline Model' : 'AI Clone'} response as better.
                `;
                chatHistory.appendChild(successMessage);
                
                // Remove success message after 3 seconds
                setTimeout(() => {
                    if (successMessage.parentNode) {
                        successMessage.remove();
                    }
                }, 3000);
                
            } catch (error) {
                console.error('Error submitting evaluation:', error);
                addMessage('Sorry, there was an error submitting your evaluation. Please try again.', 'ai');
                
                // Re-enable buttons on error
                resetVoteButtons();
            }
        });
    });
    
    // Function to reset the comparison interface
    function resetComparisonInterface() {
        // Reset progress steps
        const progressSteps = document.querySelectorAll('.progress-step');
        progressSteps.forEach((step, index) => {
            if (index === 0) {
                step.classList.add('active');
            } else {
                step.classList.remove('active');
            }
        });
        
        // Clear response areas
        const baselineResponse = document.getElementById('baseline-response');
        const langchainResponse = document.getElementById('langchain-response');
        
        if (baselineResponse) {
            baselineResponse.innerHTML = `
                <div class="response-loading">
                    <div class="loading-spinner"></div>
                    <span>Generating baseline response...</span>
                </div>
            `;
        }
        
        if (langchainResponse) {
            langchainResponse.innerHTML = `
                <div class="response-loading">
                    <div class="loading-spinner"></div>
                    <span>Generating AI clone response...</span>
                </div>
            `;
        }
        
        // Reset vote buttons
        resetVoteButtons();
        
        // Clear current question
        const currentQuestionElement = document.getElementById('current-question');
        if (currentQuestionElement) {
            currentQuestionElement.textContent = '';
        }
    }
    
    // Function to reset vote buttons
    function resetVoteButtons() {
        const voteButtons = document.querySelectorAll('.vote-button');
        voteButtons.forEach(btn => {
            btn.disabled = true;
            btn.style.opacity = '1';
            
            // Reset baseline button
            if (btn.classList.contains('baseline-vote')) {
                btn.style.backgroundColor = '';
                const span = btn.querySelector('span');
                if (span) span.textContent = 'Choose Baseline';
            }
            
            // Reset clone button
            if (btn.classList.contains('clone-vote')) {
                btn.style.backgroundColor = '';
                const span = btn.querySelector('span');
                if (span) span.textContent = 'Choose AI Clone';
            }
        });
    }
    
    // Add suggestion chip functionality
    const suggestionChips = document.querySelectorAll('.suggestion-chip');
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', function() {
            const question = this.getAttribute('data-question');
            if (userInput && question) {
                userInput.value = question;
                // Trigger send automatically or just fill the input
                userInput.focus();
            }
        });
    });
    
    // Add character counter functionality
    if (userInput) {
        const charCounter = document.getElementById('char-count');
        if (charCounter) {
            userInput.addEventListener('input', function() {
                const currentLength = this.value.length;
                charCounter.textContent = currentLength;
                
                // Add visual feedback for character limit
                if (currentLength > 450) {
                    charCounter.style.color = '#ef4444'; // Red
                } else if (currentLength > 400) {
                    charCounter.style.color = '#f59e0b'; // Amber
                } else {
                    charCounter.style.color = '#64748b'; // Default
                }
            });
        }
    }
    
    // Add action button functionality
    const skipButton = document.getElementById('skip-evaluation');
    const restartButton = document.getElementById('restart-evaluation');
    
    if (skipButton) {
        skipButton.addEventListener('click', function() {
            const comparisonContainer = document.getElementById('comparison-container');
            if (comparisonContainer) {
                comparisonContainer.style.display = 'none';
                resetComparisonInterface();
                addMessage('Evaluation skipped. You can ask another question.', 'ai');
            }
        });
    }
    
    if (restartButton) {
        restartButton.addEventListener('click', function() {
            const comparisonContainer = document.getElementById('comparison-container');
            if (comparisonContainer) {
                comparisonContainer.style.display = 'none';
                resetComparisonInterface();
                addMessage('Ready for a new evaluation. Ask me another question!', 'ai');
            }
        });
    }

    // Function to update analysis display
    async function updateAnalysis() {
        try {
            const response = await fetch('/api/eval_analysis');
            if (!response.ok) throw new Error('Failed to fetch analysis');
            
            const data = await response.json();
            
            // Update total evaluations
            document.getElementById('total-evals').textContent = data.total_evaluations;
            
            // Update type breakdown
            const typeBreakdown = document.getElementById('type-breakdown');
            typeBreakdown.innerHTML = '';
            for (const [type, count] of Object.entries(data.type_breakdown)) {
                const typeDiv = document.createElement('div');
                typeDiv.className = 'type-stat';
                typeDiv.innerHTML = `
                    <span class="type-name">${type.replace('_', ' ').toUpperCase()}</span>
                    <span class="type-count">${count}</span>
                `;
                typeBreakdown.appendChild(typeDiv);
            }
            
            // Update model performance
            const modelPerformance = document.getElementById('model-performance');
            modelPerformance.innerHTML = '';
            for (const [type, stats] of Object.entries(data.model_performance)) {
                const perfDiv = document.createElement('div');
                perfDiv.className = 'perf-stat';
                perfDiv.innerHTML = `
                    <h5>${type.replace('_', ' ').toUpperCase()}</h5>
                    <div class="perf-bars">
                        <div class="perf-bar">
                            <div class="bar-label">Baseline</div>
                            <div class="bar-container">
                                <div class="bar" style="width: ${stats.baseline_win_rate}%"></div>
                                <span class="bar-value">${stats.baseline_win_rate}%</span>
                            </div>
                        </div>
                        <div class="perf-bar">
                            <div class="bar-label">Cloned</div>
                            <div class="bar-container">
                                <div class="bar" style="width: ${stats.langchain_win_rate}%"></div>
                                <span class="bar-value">${stats.langchain_win_rate}%</span>
                            </div>
                        </div>
                    </div>
                    <div class="total-evals">Total: ${stats.total_evaluations}</div>
                `;
                modelPerformance.appendChild(perfDiv);
            }
        } catch (error) {
            console.error('Error updating analysis:', error);
        }
    }
    
    // Initial analysis update
    updateAnalysis();
});
