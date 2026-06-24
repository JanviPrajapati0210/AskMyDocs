/* ================================================================
   RAG Chatbot — chat.js
   Handles: file upload, sendMessage (SSE streaming),
            conversation memory, citations, UI helpers
   ================================================================ */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let selectedFiles = [];
let isLoading     = false;


// ════════════════════════════════════════════════════════════════════════════
// FILE UPLOAD
// ════════════════════════════════════════════════════════════════════════════

/**
 * Called when user selects files via the file input.
 * Updates selectedFiles and renders file badge list.
 */
function handleFiles(files) {
  selectedFiles = Array.from(files);
  renderFileBadges();
}

/**
 * Drag-over handler — adds visual feedback on the upload area.
 */
function handleDragOver(event) {
  event.preventDefault();
  document.getElementById('upload-area').classList.add('drag-over');
}

/**
 * Drop handler — reads dropped files and renders badges.
 */
function handleDrop(event) {
  event.preventDefault();
  document.getElementById('upload-area').classList.remove('drag-over');
  selectedFiles = Array.from(event.dataTransfer.files);
  renderFileBadges();
}

/**
 * Renders the list of selected file badges below the upload area.
 */
function renderFileBadges() {
  const list = document.getElementById('file-list');
  list.innerHTML = selectedFiles
    .map(f => `<span class="file-badge">📄 ${escHtml(f.name)}</span>`)
    .join('');
}

/**
 * Sends selected files to POST /upload via FormData.
 * Shows animated progress bar and inline status messages.
 */
async function uploadFiles() {
  if (!selectedFiles.length) {
    showUploadError('Please select at least one file first.');
    return;
  }

  const btn      = document.getElementById('upload-btn');
  const progress = document.getElementById('upload-progress');
  const bar      = document.getElementById('upload-bar');
  const errorEl  = document.getElementById('upload-error');
  const statusEl = document.getElementById('upload-status');

  // Reset UI state
  errorEl.style.display  = 'none';
  statusEl.textContent   = '';
  statusEl.className     = 'upload-status';
  btn.disabled           = true;
  btn.textContent        = 'Processing...';
  progress.style.display = 'block';
  bar.style.width        = '10%';

  const formData = new FormData();
  selectedFiles.forEach(f => formData.append('file', f));

  try {
    bar.style.width = '40%';

    const res  = await fetch('/upload', { method: 'POST', body: formData });
    bar.style.width = '80%';

    const data = await res.json();

    if (data.error) {
      showUploadError(data.error);
      bar.style.width = '0%';
      return;
    }

    // Success
    bar.style.width = '100%';
    setUploadStatus(data.message, 'ok');
    removeEmptyState();

    addBotMessage(
      `Documents ready! I've processed <strong>${data.files.join(', ')}</strong> ` +
      `into <strong>${data.chunks}</strong> searchable chunks. Ask me anything!`,
      []
    );

    // Hide progress bar after a short delay
    setTimeout(() => {
      progress.style.display = 'none';
      bar.style.width        = '0%';
    }, 900);

  } catch (err) {
    showUploadError('Upload failed — is the server running?');
    bar.style.width        = '0%';
    progress.style.display = 'none';
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Upload & Process';
  }
}

/**
 * Shows an inline error message below the upload button.
 */
function showUploadError(message) {
  const el = document.getElementById('upload-error');
  el.textContent  = message;
  el.style.display = 'block';
}

/**
 * Sets the upload status text with a style class (ok / err).
 */
function setUploadStatus(message, type) {
  const el = document.getElementById('upload-status');
  el.textContent = message;
  el.className   = `upload-status ${type}`;
}


// ════════════════════════════════════════════════════════════════════════════
// CHAT — STREAMING (SSE)
// ════════════════════════════════════════════════════════════════════════════

/**
 * Main send function — opens an SSE connection to GET /stream
 * and streams tokens word-by-word into the chat bubble.
 */
async function sendMessage() {
  const input = document.getElementById('question');
  const sendBtn = document.getElementById('send-btn');
  const question = input.value.trim();

  if (!question || isLoading) return;

  isLoading = true;
  input.value = '';
  input.style.height = 'auto';

  sendBtn.disabled = true;
  sendBtn.innerHTML = '<div class="spinner"></div>';

  removeEmptyState();
  addUserMessage(question);

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        question: question
      })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Server error');
    }

    addBotMessage(
      data.answer,
      data.sources || []
    );

  } catch (err) {
    console.error(err);
    addErrorMessage(err.message || 'Connection error');
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    sendBtn.innerHTML = '➤';
    input.focus();
  }
}

// ════════════════════════════════════════════════════════════════════════════
// CHAT — MEMORY ACTIONS
// ════════════════════════════════════════════════════════════════════════════

/**
 * Resets conversation memory on the server without clearing documents.
 * Shows a fresh welcome message in the chat.
 */
async function newChat() {
  try {
    await fetch('/new-chat', { method: 'POST' });

    const msgs = document.getElementById('messages');
    msgs.innerHTML = '';
    addBotMessage(
      'New conversation started! Your documents are still loaded — ask me anything.',
      []
    );
  } catch (err) {
    console.error('Failed to start new chat:', err);
  }
}

/**
 * Clears all documents, ChromaDB, uploads, and conversation memory.
 * Resets the UI to the initial empty state.
 */
async function resetAll() {
  if (!confirm('This will delete all uploaded documents and clear the chat. Continue?')) return;

  try {
    await fetch('/reset', { method: 'POST' });

    // Reset message area
    const msgs = document.getElementById('messages');
    msgs.innerHTML = `
      <div class="empty-state" id="empty-state">
        <div class="empty-icon">📄</div>
        <p class="empty-title">No documents loaded</p>
        <p class="empty-hint">Upload a PDF, DOCX, or TXT file to get started</p>
      </div>`;

    // Reset sidebar
    document.getElementById('file-list').innerHTML = '';
    setUploadStatus('All documents cleared.', 'ok');

  } catch (err) {
    console.error('Reset failed:', err);
  }
}


// ════════════════════════════════════════════════════════════════════════════
// UI HELPERS — MESSAGES
// ════════════════════════════════════════════════════════════════════════════

/**
 * Appends a user message bubble to the chat.
 */
function addUserMessage(text) {
  const msgs = document.getElementById('messages');
  const el   = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `
    <div class="avatar">J</div>
    <div class="bubble">${escHtml(text)}</div>`;
  msgs.appendChild(el);
  scrollToBottom();
}

/**
 * Appends a completed bot message bubble with optional citation cards.
 * Used for non-streamed messages (upload confirmation, new chat, etc.)
 */
function addBotMessage(html, sources) {
  const msgs = document.getElementById('messages');
  const el   = document.createElement('div');
  el.className = 'msg bot';

  const citationsHtml = buildCitationsHtml(sources);

  el.innerHTML = `
    <div class="avatar">🤖</div>
    <div>
      <div class="bubble">${html}</div>
      ${citationsHtml}
    </div>`;
  msgs.appendChild(el);
  scrollToBottom();
}

/**
 * Appends a red error bubble to the chat.
 */
function addErrorMessage(text) {
  const msgs = document.getElementById('messages');
  const el   = document.createElement('div');
  el.className = 'msg bot';
  el.innerHTML = `
    <div class="avatar" style="background:#FEE2E2;">⚠️</div>
    <div class="bubble error">${escHtml(text)}</div>`;
  msgs.appendChild(el);
  scrollToBottom();
}

/**
 * Creates an empty bot bubble with a blinking cursor for streaming.
 * The bubble is identified by msgId so it can be updated mid-stream.
 */
function createStreamBubble(msgId) {
  const msgs = document.getElementById('messages');
  const el   = document.createElement('div');
  el.className = 'msg bot';
  el.id        = msgId;
  el.innerHTML = `
    <div class="avatar">🤖</div>
    <div>
      <div class="bubble" id="${msgId}-text">
        <span class="cursor"></span>
      </div>
    </div>`;
  msgs.appendChild(el);
  scrollToBottom();
}

/**
 * Updates the streaming bubble with accumulated text.
 * @param {string}  msgId  - bubble element ID
 * @param {string}  text   - full accumulated text so far
 * @param {boolean} done   - if true, remove cursor and finalise
 */
function updateStreamBubble(msgId, text, done) {
  const el = document.getElementById(`${msgId}-text`);
  if (!el) return;

  if (done) {
    el.innerHTML = escHtml(text); // finalise — no cursor
  } else {
    el.innerHTML = escHtml(text) + '<span class="cursor"></span>';
  }
  scrollToBottom();
}

/**
 * Appends citation cards below a finished streaming bubble.
 * @param {string} msgId   - ID of the parent message element
 * @param {Array}  sources - array of {filename, page, excerpt} dicts
 */
function appendCitations(msgId, sources) {
  const msgEl    = document.getElementById(msgId);
  if (!msgEl) return;

  const innerDiv = msgEl.querySelector('div:last-child');
  if (!innerDiv) return;

  innerDiv.insertAdjacentHTML('beforeend', buildCitationsHtml(sources));
  scrollToBottom();
}

/**
 * Builds the HTML string for citation cards.
 * Handles both new dict format {filename, page, excerpt}
 * and old string format (plain filename string).
 */
function buildCitationsHtml(sources) {
  if (!sources || sources.length === 0) return '';

  const cards = sources.map(s => {
    // Backward-compatible: handle plain strings
    if (typeof s === 'string') {
      return `<div class="citation-card">
        <div class="citation-top">
          <span class="citation-filename">📄 ${escHtml(s)}</span>
        </div>
      </div>`;
    }

    const pageBadge = s.page
      ? `<span class="citation-page">Page ${s.page}</span>`
      : '';

    const excerpt = s.excerpt
      ? `<div class="citation-excerpt">"${escHtml(s.excerpt)}"</div>`
      : '';

    return `<div class="citation-card">
      <div class="citation-top">
        <span class="citation-filename">📄 ${escHtml(s.filename)}</span>
        ${pageBadge}
      </div>
      ${excerpt}
    </div>`;
  }).join('');

  return `
    <div class="citations">
      <div class="citations-label">Sources</div>
      ${cards}
    </div>`;
}


// ════════════════════════════════════════════════════════════════════════════
// UI HELPERS — GENERAL
// ════════════════════════════════════════════════════════════════════════════

/**
 * Sets the question textarea value and focuses it.
 * Called by suggestion buttons.
 */
function setQuestion(text) {
  const input = document.getElementById('question');
  input.value = text;
  input.focus();
  autoResize(input);
}

/**
 * Handles keydown on the textarea.
 * Enter = send, Shift+Enter = newline.
 */
function handleKey(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

/**
 * Auto-resizes the textarea as the user types.
 * Caps at 120px height.
 */
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

/**
 * Scrolls the message list to the bottom.
 */
function scrollToBottom() {
  const msgs = document.getElementById('messages');
  msgs.scrollTop = msgs.scrollHeight;
}

/**
 * Removes the empty state placeholder if it exists.
 */
function removeEmptyState() {
  document.getElementById('empty-state')?.remove();
}

/**
 * Escapes HTML special characters to prevent XSS.
 * Also converts **bold** markdown to <strong> tags.
 */
function escHtml(str) {
  if (typeof str !== 'string') return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}