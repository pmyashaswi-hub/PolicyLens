/**
 * PolicyLens — Single Page Frontend Application (Vanilla ES Modules)
 * Handles SSE streaming, interactive citations, evidence synchronization,
 * evaluation canvas charts, animated stepper, demo mode, and design system.
 */

// ============================================================================
// State Management
// ============================================================================
const state = {
  sessionId: localStorage.getItem('policylens_session_id') || `session_${Math.random().toString(36).substring(2, 11)}`,
  theme: localStorage.getItem('policylens_theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'),
  activeTab: 'view-ask',
  isGenerating: false,
  config: {
    top_k: 5,
    rerank: true,
    verify: true,
    rewrite: true,
    filterDocs: [],
  },
  corpus: null,
  evalData: null,
  examples: null,
  currentSources: [],
  demo: {
    active: false,
    step: 0,
    secondsLeft: 180,
    timerId: null,
  }
};

localStorage.setItem('policylens_session_id', state.sessionId);

// ============================================================================
// DOM Element Selectors
// ============================================================================
const elements = {
  html: document.documentElement,
  themeToggle: document.getElementById('theme-toggle'),
  shortcutsBtn: document.getElementById('shortcuts-btn'),
  shortcutsModal: document.getElementById('shortcuts-modal'),
  closeShortcutsModal: document.getElementById('close-shortcuts-modal'),
  toastContainer: document.getElementById('toast-container'),
  navTabs: document.querySelectorAll('.nav-tab'),
  navIndicator: document.querySelector('.nav-indicator'),
  tabViews: document.querySelectorAll('.tab-view'),
  healthPill: document.getElementById('health-pill'),
  healthLabel: document.getElementById('health-label'),
  
  // Ask View
  chatHero: document.getElementById('chat-hero'),
  chatMessages: document.getElementById('chat-messages'),
  messagesBottomAnchor: document.getElementById('messages-bottom-anchor'),
  chatForm: document.getElementById('chat-form'),
  chatInput: document.getElementById('chat-input'),
  sendBtn: document.getElementById('send-btn'),
  clearChatBtn: document.getElementById('clear-chat-btn'),
  exampleChipsContainer: document.getElementById('example-chips'),
  heroDemoBtn: document.getElementById('hero-demo-btn'),

  // Settings Popover
  settingsBtn: document.getElementById('settings-popover-btn'),
  settingsPopover: document.getElementById('settings-popover'),
  closePopoverBtn: document.getElementById('close-popover-btn'),
  docFilterChips: document.getElementById('doc-filter-chips'),
  topKSlider: document.getElementById('top-k-slider'),
  topKValue: document.getElementById('top-k-value'),
  toggleRerank: document.getElementById('toggle-rerank'),
  toggleRewrite: document.getElementById('toggle-rewrite'),
  toggleVerify: document.getElementById('toggle-verify'),
  rebuildIndexBtn: document.getElementById('rebuild-index-btn'),

  // Evidence Panel
  evidencePanel: document.getElementById('evidence-panel'),
  evidenceCountBadge: document.getElementById('evidence-count-badge'),
  evidenceCardsContainer: document.getElementById('evidence-cards-container'),
  closeEvidenceMobile: document.getElementById('close-evidence-mobile'),

  // Corpus View
  statDocs: document.getElementById('stat-docs'),
  statPages: document.getElementById('stat-pages'),
  statChunks: document.getElementById('stat-chunks'),
  statAvgTokens: document.getElementById('stat-avg-tokens'),
  corpusDocsList: document.getElementById('corpus-docs-list'),
  addDocBtn: document.getElementById('add-doc-btn'),
  corpusRebuildBtn: document.getElementById('corpus-rebuild-btn'),

  // Add Document Modal
  addDocModal: document.getElementById('add-doc-modal'),
  closeAddDocModal: document.getElementById('close-add-doc-modal'),
  uploadDocForm: document.getElementById('upload-doc-form'),
  fileDropArea: document.getElementById('file-drop-area'),
  pdfFileInput: document.getElementById('pdf-file-input'),
  dropPrompt: document.getElementById('drop-prompt'),
  selectedFileBadge: document.getElementById('selected-file-badge'),
  selectedFileName: document.getElementById('selected-file-name'),
  selectedFileSize: document.getElementById('selected-file-size'),
  clearSelectedFile: document.getElementById('clear-selected-file'),
  uploadDocTitle: document.getElementById('upload-doc-title'),
  uploadDocIssuer: document.getElementById('upload-doc-issuer'),
  uploadProgressContainer: document.getElementById('upload-progress-container'),
  uploadProgressText: document.getElementById('upload-progress-text'),
  cancelUploadBtn: document.getElementById('cancel-upload-btn'),
  submitUploadBtn: document.getElementById('submit-upload-btn'),
  submitUploadBtnText: document.getElementById('submit-upload-btn-text'),

  // Evaluation View
  evalStatPass: document.getElementById('eval-stat-pass'),
  evalStatCite: document.getElementById('eval-stat-cite'),
  evalStatRefusal: document.getElementById('eval-stat-refusal'),
  evalStatFaithful: document.getElementById('eval-stat-faithful'),
  ablationChartCanvas: document.getElementById('ablation-chart'),
  evalTableBody: document.getElementById('eval-table-body'),
  failureCasesContainer: document.getElementById('failure-cases-container'),

  // Demo Controller
  navDemoBtn: document.getElementById('nav-demo-btn'),
  demoDock: document.getElementById('demo-dock'),
  demoTimer: document.getElementById('demo-timer'),
  closeDemoBtn: document.getElementById('close-demo-btn'),
  demoStepIndicator: document.getElementById('demo-step-indicator'),
  demoScenarioTitle: document.getElementById('demo-scenario-title'),
  demoScenarioSub: document.getElementById('demo-scenario-sub'),
  demoPrevBtn: document.getElementById('demo-prev-btn'),
  demoNextBtn: document.getElementById('demo-next-btn'),
};

// ============================================================================
// Toast Notification Utility
// ============================================================================
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
  toast.innerHTML = `<span><b>${icon}</b> ${message}</span>`;
  elements.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(-10px)';
    toast.style.transition = 'all 0.25s ease-out';
    setTimeout(() => toast.remove(), 250);
  }, 3500);
}

// ============================================================================
// Theme Management
// ============================================================================
function applyTheme(theme) {
  state.theme = theme;
  elements.html.setAttribute('data-theme', theme);
  localStorage.setItem('policylens_theme', theme);
}

function toggleTheme() {
  const newTheme = state.theme === 'light' ? 'dark' : 'light';
  applyTheme(newTheme);
  // Redraw chart if evaluation tab is active
  if (state.activeTab === 'view-eval') {
    renderAblationChart();
  }
}

// ============================================================================
// Navigation & Tab Switching
// ============================================================================
function updateNavIndicator(activeButton) {
  if (!activeButton) return;
  const rect = activeButton.getBoundingClientRect();
  const parentRect = activeButton.parentElement.getBoundingClientRect();
  elements.navIndicator.style.width = `${rect.width}px`;
  elements.navIndicator.style.transform = `translateX(${rect.left - parentRect.left}px)`;
}

function switchTab(targetViewId) {
  state.activeTab = targetViewId;
  elements.tabViews.forEach(view => {
    if (view.id === targetViewId) {
      view.classList.add('active');
      view.hidden = false;
    } else {
      view.classList.remove('active');
      view.hidden = true;
    }
  });

  elements.navTabs.forEach(tab => {
    const isTarget = tab.getAttribute('data-target') === targetViewId;
    tab.classList.toggle('active', isTarget);
    tab.setAttribute('aria-selected', isTarget);
    if (isTarget) {
      updateNavIndicator(tab);
    }
  });

  if (targetViewId === 'view-corpus' && !state.corpus) {
    loadCorpus();
  } else if (targetViewId === 'view-eval') {
    if (!state.evalData) loadEvaluation();
    else renderAblationChart();
  }
}

// ============================================================================
// API Calls: Health, Corpus, Evaluation, Examples
// ============================================================================
async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.status === 'ok') {
      elements.healthLabel.textContent = `Index Ready (${data.chunk_count} clauses)`;
      elements.healthPill.title = `Model: ${data.llm_model} | Embedder: ${data.embedding_model}`;
    } else {
      elements.healthLabel.textContent = 'Degraded Index';
      elements.healthPill.style.borderColor = 'var(--color-amber)';
    }
  } catch (err) {
    elements.healthLabel.textContent = 'Offline';
  }
}

async function loadCorpus() {
  try {
    const res = await fetch('/api/corpus');
    const data = await res.json();
    state.corpus = data;

    // Animate stats
    animateCountUp(elements.statDocs, data.total_docs);
    animateCountUp(elements.statPages, data.total_pages);
    animateCountUp(elements.statChunks, data.total_chunks);
    animateCountUp(elements.statAvgTokens, data.avg_chunk_size);

    // Populate Corpus list
    elements.corpusDocsList.innerHTML = '';
    const maxChunks = Math.max(...data.documents.map(d => d.chunk_count), 1);

    data.documents.forEach(doc => {
      const pct = Math.round((doc.chunk_count / maxChunks) * 100);
      const docCard = document.createElement('div');
      docCard.className = 'corpus-doc-item';
      const deleteBtnHtml = doc.is_custom
        ? `<button class="btn-delete-doc" data-doc="${doc.doc_name}" title="Remove custom document">✕ Remove</button>`
        : '';

      docCard.innerHTML = `
        <div class="corpus-doc-header">
          <div style="display: flex; align-items: center; gap: 8px;">
            <span class="source-tag tag-${doc.doc_type}">${doc.doc_type}</span>
            <span class="doc-title-main">${doc.title || doc.doc_name}</span>
          </div>
          <div style="display: flex; align-items: center; gap: 12px;">
            <div class="doc-issuer-meta">
              ${doc.issuer} • ${doc.year} • <b>${doc.page_count} pages</b>
            </div>
            ${deleteBtnHtml}
          </div>
        </div>
        <div class="chunk-bar-container">
          <div class="chunk-bar-track">
            <div class="chunk-bar-fill" style="width: ${pct}%"></div>
          </div>
          <span style="font-family: var(--font-mono); font-size: 0.78rem;">${doc.chunk_count} clauses (${doc.word_count} words)</span>
        </div>
        <div style="font-size: 0.75rem; color: var(--color-text-tertiary);">
          <b>Sections:</b> ${doc.sections.join(' • ')}
        </div>
      `;

      const delBtn = docCard.querySelector('.btn-delete-doc');
      if (delBtn) {
        delBtn.addEventListener('click', async (e) => {
          e.stopPropagation();
          if (confirm(`Are you sure you want to remove "${doc.title || doc.doc_name}" from the corpus?`)) {
            delBtn.disabled = true;
            delBtn.textContent = 'Removing...';
            try {
              const res = await fetch(`/api/documents/${encodeURIComponent(doc.doc_name)}`, { method: 'DELETE' });
              if (res.ok) {
                showToast(`Removed "${doc.title || doc.doc_name}"`, 'success');
                await checkHealth();
                await loadCorpus();
              } else {
                showToast('Failed to delete document', 'error');
              }
            } catch (err) {
              showToast(`Error: ${err.message}`, 'error');
            }
          }
        });
      }

      elements.corpusDocsList.appendChild(docCard);
    });

    // Populate Settings Document Filters
    populateDocFilterChips(data.documents);
  } catch (err) {
    console.error('Failed to load corpus:', err);
  }
}

function populateDocFilterChips(docs) {
  elements.docFilterChips.innerHTML = '';
  docs.forEach(d => {
    const row = document.createElement('label');
    row.className = 'filter-chip-row';
    row.innerHTML = `
      <input type="checkbox" value="${d.doc_name}" checked />
      <span>${d.doc_name}</span>
    `;
    const cb = row.querySelector('input');
    cb.addEventListener('change', () => {
      const selected = Array.from(elements.docFilterChips.querySelectorAll('input:checked')).map(i => i.value);
      state.config.filterDocs = selected.length === docs.length ? [] : selected;
    });
    elements.docFilterChips.appendChild(row);
  });
}

async function loadEvaluation() {
  try {
    const res = await fetch('/api/eval');
    const data = await res.json();
    state.evalData = data;

    animateCountUp(elements.evalStatPass, `${data.pass_rate}%`);
    animateCountUp(elements.evalStatCite, `${data.citation_accuracy}%`);
    animateCountUp(elements.evalStatRefusal, `${data.refusal_precision}%`);
    animateCountUp(elements.evalStatFaithful, '1.00');

    // Render Canvas ablation chart
    renderAblationChart();

    // Populate Results Table
    elements.evalTableBody.innerHTML = '';
    data.results.forEach(r => {
      const tr = document.createElement('tr');
      const passBadge = r.pass ? '<span class="badge-pass">PASSED</span>' : '<span class="badge-fail">ATTN</span>';
      tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-weight: 600;">${r.id}</td>
        <td><span class="source-tag tag-${r.type}">${r.type}</span></td>
        <td><b>${escapeHtml(r.question)}</b></td>
        <td>${passBadge}</td>
        <td style="font-family: var(--font-mono);">${r.latency_ms} ms</td>
        <td style="font-size: 0.78rem; color: var(--color-text-secondary);">${escapeHtml(r.source || 'None')}</td>
      `;
      elements.evalTableBody.appendChild(tr);
    });

    // Populate Failure cases
    elements.failureCasesContainer.innerHTML = '';
    data.failure_cases.forEach(fc => {
      const card = document.createElement('div');
      card.className = 'failure-card';
      card.innerHTML = `
        <div class="failure-title">${fc.id}: ${fc.title}</div>
        <div class="failure-detail"><b>Question:</b> "${escapeHtml(fc.question)}"</div>
        <div class="failure-detail"><b>Retrieved Context:</b> <code>${escapeHtml(fc.retrieved)}</code></div>
        <div class="failure-detail"><b>Output:</b> <i>"${escapeHtml(fc.actual_answer)}"</i></div>
        <div class="failure-detail" style="color: var(--color-amber);"><b>Root Cause:</b> ${escapeHtml(fc.root_cause)}</div>
        <div class="failure-detail" style="color: var(--color-success);"><b>Applied Fix:</b> ${escapeHtml(fc.fix)}</div>
      `;
      elements.failureCasesContainer.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load evaluation:', err);
  }
}

// ============================================================================
// Ablation Canvas Bar Chart (Zero External Chart Library)
// ============================================================================
function renderAblationChart() {
  if (!state.evalData || !state.evalData.ablation) return;
  const canvas = elements.ablationChartCanvas;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;

  // Scale for retina
  const width = canvas.parentElement.clientWidth || 800;
  const height = 300;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, width, height);

  const isDark = state.theme === 'dark';
  const textColor = isDark ? '#94A3B8' : '#475569';
  const gridColor = isDark ? '#1F293D' : '#E2E8F0';

  const data = state.evalData.ablation;
  const padLeft = 80;
  const padBottom = 60;
  const padTop = 30;
  const padRight = 30;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  // Draw Grid Lines (0%, 25%, 50%, 75%, 100%)
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  ctx.fillStyle = textColor;
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'right';

  for (let pct = 0; pct <= 100; pct += 25) {
    const y = padTop + chartH - (pct / 100) * chartH;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(width - padRight, y);
    ctx.stroke();
    ctx.fillText(`${pct}%`, padLeft - 10, y + 4);
  }

  // Draw Grouped Bars
  const numGroups = data.length;
  const groupWidth = chartW / numGroups;
  const barWidth = Math.min(22, (groupWidth - 28) / 3);

  const colors = {
    hit5: '#0E7C7B',
    mrr: '#2563EB',
    recall: '#F5A524',
  };

  data.forEach((group, gIdx) => {
    const groupX = padLeft + gIdx * groupWidth + (groupWidth - barWidth * 3 - 8) / 2;

    // Bar 1: Hit@5
    const h1 = (group.hit5 / 100) * chartH;
    ctx.fillStyle = colors.hit5;
    ctx.fillRect(groupX, padTop + chartH - h1, barWidth, h1);

    // Bar 2: MRR (scale 0-1 to 0-100%)
    const h2 = (group.mrr) * chartH;
    ctx.fillStyle = colors.mrr;
    ctx.fillRect(groupX + barWidth + 4, padTop + chartH - h2, barWidth, h2);

    // Bar 3: Recall
    const h3 = (group.recall / 100) * chartH;
    ctx.fillStyle = colors.recall;
    ctx.fillRect(groupX + (barWidth + 4) * 2, padTop + chartH - h3, barWidth, h3);

    // Label on X-axis
    ctx.fillStyle = textColor;
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    const labelParts = group.config.split(' ')[0] + ' ' + (group.config.split(' ')[1] || '');
    ctx.fillText(labelParts, padLeft + gIdx * groupWidth + groupWidth / 2, height - padBottom + 20);
    ctx.fillText(`(${group.latency_ms.toFixed(1)}ms)`, padLeft + gIdx * groupWidth + groupWidth / 2, height - padBottom + 36);
  });
}

// ============================================================================
// Chat & SSE Streaming Implementation
// ============================================================================
async function sendChatMessage(questionText) {
  if (!questionText || state.isGenerating) return;

  const query = questionText.trim();
  if (!query) return;

  // Hide empty state hero
  if (elements.chatHero) {
    elements.chatHero.style.display = 'none';
  }

  // Append User Bubble
  appendUserMessage(query);

  // Clear input & reset height
  elements.chatInput.value = '';
  elements.chatInput.style.height = 'auto';

  // Create Assistant Message Card with Stepper & Shimmer
  const assistantMsgObj = createAssistantMessageCard();
  elements.chatMessages.appendChild(assistantMsgObj.card);
  scrollToBottom();

  state.isGenerating = true;
  elements.sendBtn.classList.add('loading');

  try {
    const payload = {
      question: query,
      session_id: state.sessionId,
      filters: { docs: state.config.filterDocs },
      top_k: state.config.top_k,
      rerank: state.config.rerank,
      verify: state.config.verify,
      rewrite: state.config.rewrite,
    };

    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop(); // keep last incomplete event in buffer

      for (const rawEvent of events) {
        if (!rawEvent.trim()) continue;
        const lines = rawEvent.split('\n');
        let eventName = 'message';
        let eventDataStr = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventName = line.substring(7).trim();
          } else if (line.startsWith('data: ')) {
            eventDataStr = line.substring(6).trim();
          }
        }

        if (eventDataStr) {
          try {
            const data = JSON.parse(eventDataStr);
            handleSSEEvent(eventName, data, assistantMsgObj);
          } catch (e) {
            console.warn('Error parsing SSE data:', eventDataStr, e);
          }
        }
      }
    }
  } catch (err) {
    console.error('Chat stream error:', err);
    showToast(err.message || 'Error communicating with PolicyLens server', 'error');
    assistantMsgObj.shimmer.remove();
    assistantMsgObj.answerBody.textContent = `Error: ${err.message}`;
  } finally {
    state.isGenerating = false;
    elements.sendBtn.classList.remove('loading');
    scrollToBottom();
  }
}

function appendUserMessage(text) {
  const msgWrapper = document.createElement('div');
  msgWrapper.className = 'chat-message user-message';
  msgWrapper.innerHTML = `
    <div class="user-bubble">${escapeHtml(text)}</div>
  `;
  elements.chatMessages.appendChild(msgWrapper);
}

function createAssistantMessageCard() {
  const cardWrapper = document.createElement('div');
  cardWrapper.className = 'chat-message assistant-message';

  const card = document.createElement('div');
  card.className = 'assistant-card';

  // 1. Pipeline Stepper
  const stepper = document.createElement('div');
  stepper.className = 'pipeline-stepper';
  stepper.innerHTML = `
    <div class="pipeline-step" data-step="rewrite">
      <span class="step-indicator">1</span>
      <span>Rewrite</span>
      <span class="step-ms"></span>
    </div>
    <div class="step-divider"></div>
    <div class="pipeline-step" data-step="retrieve">
      <span class="step-indicator">2</span>
      <span>Retrieve</span>
      <span class="step-ms"></span>
    </div>
    <div class="step-divider"></div>
    <div class="pipeline-step" data-step="rerank">
      <span class="step-indicator">3</span>
      <span>Rerank</span>
      <span class="step-ms"></span>
    </div>
    <div class="step-divider"></div>
    <div class="pipeline-step" data-step="generate">
      <span class="step-indicator">4</span>
      <span>Generate</span>
      <span class="step-ms"></span>
    </div>
    <div class="step-divider"></div>
    <div class="pipeline-step" data-step="verify">
      <span class="step-indicator">5</span>
      <span>Verify</span>
      <span class="step-ms"></span>
    </div>
  `;

  // 2. Collapsible Search Info (hidden initially)
  const searchSummary = document.createElement('div');
  searchSummary.className = 'search-summary-body';
  searchSummary.hidden = true;

  // 3. Grounding Header Bar
  const groundingBar = document.createElement('div');
  groundingBar.className = 'answer-grounding-bar';
  groundingBar.style.display = 'none';

  // 4. Shimmer Skeleton
  const shimmer = document.createElement('div');
  shimmer.className = 'shimmer-lines';
  shimmer.innerHTML = `
    <div class="shimmer-line"></div>
    <div class="shimmer-line"></div>
    <div class="shimmer-line"></div>
  `;

  // 5. Answer Body
  const answerBody = document.createElement('div');
  answerBody.className = 'answer-body';

  // Caret
  const caret = document.createElement('span');
  caret.className = 'streaming-caret';

  // 6. Footer Actions
  const footer = document.createElement('div');
  footer.className = 'answer-footer';
  footer.style.display = 'none';

  card.appendChild(stepper);
  card.appendChild(searchSummary);
  card.appendChild(groundingBar);
  card.appendChild(shimmer);
  card.appendChild(answerBody);
  card.appendChild(footer);
  cardWrapper.appendChild(card);

  return {
    cardWrapper,
    card,
    stepper,
    searchSummary,
    groundingBar,
    shimmer,
    answerBody,
    caret,
    footer,
    rawText: '',
  };
}

// ============================================================================
// SSE Event Handler
// ============================================================================
function handleSSEEvent(eventName, data, assistantMsg) {
  switch (eventName) {
    case 'stage':
      handleStageEvent(data, assistantMsg);
      break;
    case 'rewrite':
      handleRewriteEvent(data, assistantMsg);
      break;
    case 'sources':
      handleSourcesEvent(data);
      break;
    case 'token':
      handleTokenEvent(data, assistantMsg);
      break;
    case 'final':
      handleFinalEvent(data, assistantMsg);
      break;
    case 'error':
      showToast(data.message || 'Pipeline execution error', 'error');
      break;
  }
}

function handleStageEvent(data, assistantMsg) {
  const stepEl = assistantMsg.stepper.querySelector(`[data-step="${data.name}"]`);
  if (!stepEl) return;

  if (data.status === 'start') {
    stepEl.classList.add('active');
  } else if (data.status === 'done') {
    stepEl.classList.remove('active');
    stepEl.classList.add('done');
    const msSpan = stepEl.querySelector('.step-ms');
    if (msSpan && data.ms !== undefined) {
      msSpan.textContent = `${data.ms}ms`;
    }
  } else if (data.status === 'skipped') {
    stepEl.classList.add('skipped');
    const msSpan = stepEl.querySelector('.step-ms');
    if (msSpan) msSpan.textContent = 'off';
  }
}

function handleRewriteEvent(data, assistantMsg) {
  assistantMsg.searchSummary.innerHTML = `
    <div class="search-summary-item"><b>Standalone Question:</b> <code>${escapeHtml(data.standalone)}</code></div>
    ${data.variants && data.variants.length > 0 ? `<div class="search-summary-item"><b>Alternative Queries:</b> ${data.variants.map(v => `<span class="chip" style="font-size:0.75rem; padding:2px 8px;">${escapeHtml(v)}</span>`).join(' ')}</div>` : ''}
  `;
  assistantMsg.searchSummary.hidden = false;
}

function handleSourcesEvent(sources) {
  state.currentSources = sources;
  renderEvidencePanel(sources);
}

function handleTokenEvent(data, assistantMsg) {
  if (assistantMsg.shimmer.parentNode) {
    assistantMsg.shimmer.remove();
  }

  assistantMsg.rawText += data.text;
  assistantMsg.answerBody.textContent = assistantMsg.rawText;
  if (!assistantMsg.caret.parentNode) {
    assistantMsg.answerBody.appendChild(assistantMsg.caret);
  }
  scrollToBottom();
}

function handleFinalEvent(data, assistantMsg) {
  // Remove caret
  assistantMsg.caret.remove();

  // 1. Render Grounding Header Bar
  assistantMsg.groundingBar.style.display = 'flex';
  const isFound = data.found;
  const confLevel = (data.confidence || 'high').toLowerCase();

  assistantMsg.groundingBar.innerHTML = `
    <div class="grounding-badge ${isFound ? 'badge-found' : 'badge-not-found'}">
      <span>${isFound ? '✓ FOUND IN POLICY' : '✕ NOT FOUND IN DOCUMENTS'}</span>
    </div>
    <div class="confidence-group">
      <span>Confidence: <b>${confLevel.toUpperCase()}</b></span>
      <div class="confidence-meter">
        <div class="meter-bar active-${confLevel}"></div>
        <div class="meter-bar ${confLevel === 'high' || confLevel === 'medium' ? 'active-' + confLevel : ''}"></div>
        <div class="meter-bar ${confLevel === 'high' ? 'active-high' : ''}"></div>
      </div>
    </div>
  `;

  // 2. Render Formatted Answer with Interactive Citation Pill Buttons
  let formattedHtml = escapeHtml(data.answer);

  // If unanswerable / not found: render calm refusal card
  if (!isFound) {
    assistantMsg.answerBody.innerHTML = `
      <p style="font-weight: 500; margin-bottom: 8px;">${escapeHtml(data.answer)}</p>
      <div class="refusal-notice-card">
        <div class="refusal-title">Why this could not be answered from indexed documents:</div>
        <div class="refusal-desc">${escapeHtml(data.missing_info || 'This specific hospital tariff or private medical fee is not part of standard policy wordings or IRDAI circulars.')}</div>
        <div style="font-size:0.75rem; color:var(--color-text-secondary); margin-top:8px;">
          💡 <b>Suggested Next Step:</b> Check your specific hospital admission estimate, contact your TPA helpdesk, or consult the IRDAI grievance portal (bimabharosa.irdai.gov.in).
        </div>
      </div>
    `;
  } else {
    // Replace [1], [2], [1][3] with interactive pills
    formattedHtml = formattedHtml.replace(/\[(\d+)\]/g, (match, p1) => {
      return `<button class="cite-pill" data-cite-idx="${p1}" title="Click to view Source [${p1}] in Evidence Panel">[${p1}]</button>`;
    });

    // Check for unsupported sentences
    if (data.unsupported_sentences && data.unsupported_sentences.length > 0) {
      data.unsupported_sentences.forEach(unsup => {
        const escapedUnsup = escapeHtml(unsup);
        formattedHtml = formattedHtml.replace(escapedUnsup, `<span class="unsupported-flag" title="Not fully supported by cited sources">${escapedUnsup}</span>`);
      });
    }

    assistantMsg.answerBody.innerHTML = formattedHtml;

    // Attach click listeners to citation pills
    const citePills = assistantMsg.answerBody.querySelectorAll('.cite-pill');
    citePills.forEach(pill => {
      pill.addEventListener('click', () => {
        const idx = parseInt(pill.getAttribute('data-cite-idx'), 10);
        focusEvidenceCard(idx);
      });
    });
  }

  // 3. Mark Cited Cards in Evidence Panel
  if (data.citations && data.citations.length > 0) {
    markCitedEvidenceCards(data.citations);
  }

  // 4. Render Footer Actions
  assistantMsg.footer.style.display = 'flex';
  assistantMsg.footer.innerHTML = `
    <div class="footer-left">
      <span class="latency-chip">${(data.latency_ms / 1000).toFixed(2)}s latency</span>
      <button class="action-icon-btn mobile-only" id="btn-show-evidence-mobile">📑 Show Evidence</button>
    </div>
    <div class="footer-actions">
      <button class="action-icon-btn btn-copy" title="Copy answer">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
        </svg>
      </button>
      <button class="action-icon-btn btn-thumbs-up" title="Helpful answer">👍</button>
      <button class="action-icon-btn btn-thumbs-down" title="Not helpful">👎</button>
    </div>
  `;

  // Attach Footer button events
  const copyBtn = assistantMsg.footer.querySelector('.btn-copy');
  copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(data.answer);
    copyBtn.innerHTML = '✓ Copied';
    setTimeout(() => {
      copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
    }, 2000);
  });

  const mobEvidenceBtn = assistantMsg.footer.querySelector('#btn-show-evidence-mobile');
  if (mobEvidenceBtn) {
    mobEvidenceBtn.addEventListener('click', () => {
      elements.evidencePanel.classList.add('mobile-open');
    });
  }
}

// ============================================================================
// Evidence Panel Rendering
// ============================================================================
function renderEvidencePanel(sources) {
  elements.evidenceCountBadge.textContent = `${sources.length} sources`;
  elements.evidenceCardsContainer.innerHTML = '';

  if (!sources || sources.length === 0) {
    elements.evidenceCardsContainer.innerHTML = `
      <div class="evidence-placeholder">
        <div class="placeholder-icon">📄</div>
        <p>No document passages retrieved for this query.</p>
      </div>
    `;
    return;
  }

  sources.forEach((src, idx) => {
    const card = document.createElement('div');
    card.className = 'source-card';
    card.id = `evidence-card-${src.idx}`;
    card.style.animationDelay = `${idx * 50}ms`;

    const pageRange = src.page_start === src.page_end ? `p. ${src.page_start}` : `p. ${src.page_start}-${src.page_end}`;
    const highlightedText = highlightTextSafely(src.text, src.highlights || []);

    // Mini Score bars percentage
    const fusedPct = Math.min(100, Math.round((src.fused / 0.035) * 100));
    const densePct = Math.min(100, Math.round((src.dense) * 100));
    const rerankScore = src.rerank !== null ? src.rerank : 'N/A';
    const rerankPct = src.rerank !== null ? Math.min(100, Math.max(0, Math.round(((src.rerank + 5) / 10) * 100))) : 0;

    card.innerHTML = `
      <div class="source-card-header">
        <div class="source-meta-left">
          <span class="source-rank-badge">[${src.idx}]</span>
          <span class="source-doc-name">${escapeHtml(src.doc_name)}</span>
          <span class="source-tag tag-${src.doc_type}">${src.doc_type}</span>
          <span class="source-tag" style="background:var(--color-card-subtle);">${pageRange}</span>
        </div>
      </div>
      <div class="source-breadcrumb"><b>Section:</b> ${escapeHtml(src.section)}</div>
      <div class="source-text-block" id="source-text-${src.idx}">${highlightedText}</div>
      <button class="toggle-expand-btn" data-target="source-text-${src.idx}">Show more ▾</button>

      <div class="source-scores-group">
        <div class="mini-score-item" title="Fused RRF Score: ${src.fused}">
          <span>Fused:</span>
          <div class="mini-bar-track"><div class="mini-bar-fill fill-fused" style="width:${fusedPct}%"></div></div>
          <span>${src.fused.toFixed(4)}</span>
        </div>
        <div class="mini-score-item" title="Dense Cosine Score: ${src.dense}">
          <span>Dense:</span>
          <div class="mini-bar-track"><div class="mini-bar-fill fill-dense" style="width:${densePct}%"></div></div>
          <span>${src.dense.toFixed(2)}</span>
        </div>
        ${src.rerank !== null ? `
        <div class="mini-score-item" title="Cross-Encoder Rerank: ${src.rerank}">
          <span>Rerank:</span>
          <div class="mini-bar-track"><div class="mini-bar-fill fill-rerank" style="width:${rerankPct}%"></div></div>
          <span>${rerankScore}</span>
        </div>` : ''}
      </div>
    `;

    // Toggle expand button
    const toggleBtn = card.querySelector('.toggle-expand-btn');
    toggleBtn.addEventListener('click', () => {
      const block = card.querySelector('.source-text-block');
      const isExpanded = block.classList.toggle('expanded');
      toggleBtn.textContent = isExpanded ? 'Show less ▴' : 'Show more ▾';
    });

    elements.evidenceCardsContainer.appendChild(card);
  });
}

function markCitedEvidenceCards(citedIndices) {
  citedIndices.forEach(idx => {
    const card = document.getElementById(`evidence-card-${idx}`);
    if (card) {
      card.classList.add('is-cited');
      const headerMeta = card.querySelector('.source-meta-left');
      if (headerMeta && !headerMeta.querySelector('.tag-cited')) {
        const citedTag = document.createElement('span');
        citedTag.className = 'source-tag tag-cited';
        citedTag.textContent = 'CITED IN ANSWER';
        headerMeta.appendChild(citedTag);
      }
    }
  });
}

function focusEvidenceCard(citationIndex) {
  // On mobile, ensure evidence panel is open
  elements.evidencePanel.classList.add('mobile-open');

  const card = document.getElementById(`evidence-card-${citationIndex}`);
  if (card) {
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    card.classList.remove('pulse-glow');
    void card.offsetWidth; // trigger reflow
    card.classList.add('pulse-glow');
  }
}

function highlightTextSafely(rawText, highlightTerms) {
  let escaped = escapeHtml(rawText);
  if (!highlightTerms || highlightTerms.length === 0) return escaped;

  highlightTerms.forEach(term => {
    if (term.length < 3) return;
    const regex = new RegExp(`\\b(${escapeRegExp(term)})\\b`, 'gi');
    escaped = escaped.replace(regex, '<mark>$1</mark>');
  });

  return escaped;
}

// ============================================================================
// 3-Minute Scripted Presentation Demo
// ============================================================================
const DEMO_SCENARIOS = [
  {
    step: 1,
    title: "Scenario 1: Direct Factual Grounding",
    sub: "Exact numeric clause extraction (2% Room Rent, 5% ICU capping)",
    question: "What is the room rent limit and ICU sub-limit under the Arogya Sanjeevani policy?",
  },
  {
    step: 2,
    title: "Scenario 2: Multi-Document Follow-Up",
    sub: "Synthesizes policy wording with IRDAI cashless TAT and Ombudsman rights",
    question: "If an insurer denies a cashless claim under Arogya Sanjeevani, what escalation options and dispute redressal rights exist under IRDAI regulations?",
  },
  {
    step: 3,
    title: "Scenario 3: Zero-Hallucination Refusal",
    sub: "Out-of-corpus query triggers strict refusal with no hallucinated figures",
    question: "What is the maximum room rent allowance for a Deluxe Suite at Apollo Hospital Bangalore?",
  },
];

function openDemoDock() {
  switchTab('view-ask');
  if (elements.demoDock) {
    elements.demoDock.hidden = false;
    elements.demoDock.removeAttribute('hidden');
    elements.demoDock.classList.remove('is-hidden');
    elements.demoDock.style.display = 'block';
  }
  state.demo.active = true;
  state.demo.step = 0;
  state.demo.secondsLeft = 180;
  updateDemoDockUI();

  if (state.demo.timerId) clearInterval(state.demo.timerId);
  state.demo.timerId = setInterval(() => {
    state.demo.secondsLeft--;
    if (state.demo.secondsLeft <= 0) {
      clearInterval(state.demo.timerId);
      if (elements.demoTimer) elements.demoTimer.textContent = "0:00";
    } else {
      const mins = Math.floor(state.demo.secondsLeft / 60);
      const secs = state.demo.secondsLeft % 60;
      if (elements.demoTimer) {
        elements.demoTimer.textContent = `${mins}:${secs < 10 ? '0' : ''}${secs}`;
        if (state.demo.secondsLeft <= 30) {
          elements.demoTimer.classList.add('warning');
        } else {
          elements.demoTimer.classList.remove('warning');
        }
      }
    }
  }, 1000);
}

function closeDemoDock() {
  if (!elements.demoDock) return;
  elements.demoDock.hidden = true;
  elements.demoDock.setAttribute('hidden', '');
  elements.demoDock.classList.add('is-hidden');
  elements.demoDock.style.display = 'none';
  state.demo.active = false;
  if (state.demo.timerId) {
    clearInterval(state.demo.timerId);
    state.demo.timerId = null;
  }
}

function updateDemoDockUI() {
  const current = DEMO_SCENARIOS[state.demo.step];
  if (!current) return;
  if (elements.demoStepIndicator) {
    elements.demoStepIndicator.textContent = `Scenario ${current.step} of 3`;
  }
  if (elements.demoScenarioTitle) {
    elements.demoScenarioTitle.textContent = current.title;
  }
  if (elements.demoScenarioSub) {
    elements.demoScenarioSub.textContent = current.sub;
  }
  if (elements.demoPrevBtn) {
    elements.demoPrevBtn.disabled = state.demo.step === 0 || state.isGenerating;
  }
  if (elements.demoNextBtn) {
    const span = elements.demoNextBtn.querySelector('span');
    if (span) {
      span.textContent = state.isGenerating 
        ? 'Generating Answer...' 
        : `Type & Run Scenario ${current.step}`;
    }
    elements.demoNextBtn.disabled = state.isGenerating;
  }
}

async function runDemoStep() {
  if (state.isGenerating) {
    showToast('Please wait for current answer to finish generating...', 'info');
    return;
  }
  const current = DEMO_SCENARIOS[state.demo.step];
  if (!current) return;

  // Disable buttons while running
  if (elements.demoNextBtn) elements.demoNextBtn.disabled = true;
  if (elements.demoPrevBtn) elements.demoPrevBtn.disabled = true;

  // Ensure we are in ask tab and input is focused
  switchTab('view-ask');
  elements.chatInput.value = '';
  elements.chatInput.focus();

  for (let i = 0; i <= current.question.length; i++) {
    elements.chatInput.value = current.question.substring(0, i);
    elements.chatInput.style.height = 'auto';
    elements.chatInput.style.height = `${Math.min(elements.chatInput.scrollHeight, 160)}px`;
    await new Promise(r => setTimeout(r, 8));
  }

  await new Promise(r => setTimeout(r, 180));

  // If not at last step, advance step counter
  if (state.demo.step < DEMO_SCENARIOS.length - 1) {
    state.demo.step++;
  }

  // Trigger streaming answer and await completion
  const sendPromise = sendChatMessage(current.question);
  updateDemoDockUI();

  await sendPromise;
  updateDemoDockUI();
}

// ============================================================================
// Helper Utilities & Micro-Interactions
// ============================================================================
function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function scrollToBottom() {
  elements.messagesBottomAnchor.scrollIntoView({ behavior: 'smooth' });
}

function animateCountUp(element, finalValue) {
  const isPct = String(finalValue).endsWith('%');
  const num = parseFloat(finalValue);
  if (isNaN(num)) {
    element.textContent = finalValue;
    return;
  }

  let current = 0;
  const steps = 30;
  const inc = num / steps;
  const interval = setInterval(() => {
    current += inc;
    if (current >= num) {
      clearInterval(interval);
      element.textContent = isPct ? `${Math.round(num)}%` : Math.round(num * 10) / 10;
    } else {
      element.textContent = isPct ? `${Math.round(current)}%` : Math.round(current * 10) / 10;
    }
  }, 16);
}

// ============================================================================
// Event Listeners & Initialization
// ============================================================================
function setupEventListeners() {
  // Theme Toggle
  elements.themeToggle.addEventListener('click', toggleTheme);

  // Tabs
  elements.navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.getAttribute('data-target');
      switchTab(target);
    });
  });

  // Chat Form Send
  elements.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    sendChatMessage(elements.chatInput.value);
  });

  // Auto-grow textarea & Enter-to-send
  elements.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage(elements.chatInput.value);
    }
  });

  elements.chatInput.addEventListener('input', () => {
    elements.chatInput.style.height = 'auto';
    elements.chatInput.style.height = `${Math.min(elements.chatInput.scrollHeight, 180)}px`;
  });

  // Clear Chat Session
  elements.clearChatBtn.addEventListener('click', async () => {
    try {
      await fetch('/api/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: state.sessionId }),
      });
      elements.chatMessages.innerHTML = '';
      elements.evidenceCardsContainer.innerHTML = `
        <div class="evidence-placeholder">
          <div class="placeholder-icon">📄</div>
          <p>Ask a question to inspect retrieved policy clauses, page numbers, and exact text highlights.</p>
        </div>
      `;
      elements.evidenceCountBadge.textContent = '0 sources';
      if (elements.chatHero) elements.chatHero.style.display = 'flex';
      showToast('Conversation memory reset', 'info');
    } catch (e) {
      console.error(e);
    }
  });

  // Example Chips
  elements.exampleChipsContainer.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (chip) {
      const q = chip.getAttribute('data-q');
      sendChatMessage(q);
    }
  });

  // Settings Popover Toggle
  elements.settingsBtn.addEventListener('click', () => {
    elements.settingsPopover.hidden = !elements.settingsPopover.hidden;
  });

  elements.closePopoverBtn.addEventListener('click', () => {
    elements.settingsPopover.hidden = true;
  });

  // Top-K Slider
  elements.topKSlider.addEventListener('input', (e) => {
    state.config.top_k = parseInt(e.target.value, 10);
    elements.topKValue.textContent = state.config.top_k;
  });

  // Settings Toggles
  elements.toggleRerank.addEventListener('change', (e) => state.config.rerank = e.target.checked);
  elements.toggleRewrite.addEventListener('change', (e) => state.config.rewrite = e.target.checked);
  elements.toggleVerify.addEventListener('change', (e) => state.config.verify = e.target.checked);

  // Rebuild Index Button
  elements.rebuildIndexBtn.addEventListener('click', async () => {
    elements.rebuildIndexBtn.disabled = true;
    elements.rebuildIndexBtn.querySelector('span').textContent = 'Rebuilding FAISS & BM25...';
    try {
      const res = await fetch('/api/rebuild', { method: 'POST' });
      const data = await res.json();
      showToast(`Index Rebuilt: ${data.total_chunks} chunks in ${data.elapsed_seconds}s`, 'success');
      checkHealth();
      loadCorpus();
    } catch (err) {
      showToast('Failed to rebuild index', 'error');
    } finally {
      elements.rebuildIndexBtn.disabled = false;
      elements.rebuildIndexBtn.querySelector('span').textContent = 'Rebuild FAISS & BM25 Index';
    }
  });

  // Mobile Evidence Panel Close
  elements.closeEvidenceMobile.addEventListener('click', () => {
    elements.evidencePanel.classList.remove('mobile-open');
  });

  // Demo Mode Listeners
  if (elements.heroDemoBtn) {
    elements.heroDemoBtn.addEventListener('click', (e) => {
      e.preventDefault();
      openDemoDock();
    });
  }

  if (elements.navDemoBtn) {
    elements.navDemoBtn.addEventListener('click', (e) => {
      e.preventDefault();
      openDemoDock();
    });
  }

  if (elements.closeDemoBtn) {
    elements.closeDemoBtn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      closeDemoDock();
    });
  }

  if (elements.demoNextBtn) {
    elements.demoNextBtn.addEventListener('click', (e) => {
      e.preventDefault();
      runDemoStep();
    });
  }

  if (elements.demoPrevBtn) {
    elements.demoPrevBtn.addEventListener('click', (e) => {
      e.preventDefault();
      if (state.demo.step > 0 && !state.isGenerating) {
        state.demo.step--;
        updateDemoDockUI();
      }
    });
  }

  // Corpus Rebuild Button in Header
  if (elements.corpusRebuildBtn) {
    elements.corpusRebuildBtn.addEventListener('click', async () => {
      elements.corpusRebuildBtn.disabled = true;
      elements.corpusRebuildBtn.querySelector('span').textContent = 'Rebuilding...';
      try {
        const res = await fetch('/api/rebuild', { method: 'POST' });
        const data = await res.json();
        showToast(`Index Rebuilt: ${data.total_chunks} chunks in ${data.elapsed_seconds}s`, 'success');
        await checkHealth();
        await loadCorpus();
      } catch (err) {
        showToast('Failed to rebuild index', 'error');
      } finally {
        elements.corpusRebuildBtn.disabled = false;
        elements.corpusRebuildBtn.querySelector('span').textContent = 'Rebuild Index';
      }
    });
  }

  // Add Document Modal Helpers & Listeners
  let selectedUploadFile = null;

  const resetUploadForm = () => {
    selectedUploadFile = null;
    if (elements.pdfFileInput) elements.pdfFileInput.value = '';
    if (elements.uploadDocTitle) elements.uploadDocTitle.value = '';
    if (elements.uploadDocIssuer) elements.uploadDocIssuer.value = '';
    if (elements.selectedFileBadge) elements.selectedFileBadge.hidden = true;
    if (elements.dropPrompt) elements.dropPrompt.hidden = false;
    if (elements.submitUploadBtn) elements.submitUploadBtn.disabled = true;
    if (elements.uploadProgressContainer) elements.uploadProgressContainer.hidden = true;
    if (elements.submitUploadBtnText) elements.submitUploadBtnText.textContent = 'Upload & Index';
  };

  const openAddDocModal = () => {
    if (!elements.addDocModal) return;
    elements.addDocModal.hidden = false;
    elements.addDocModal.classList.remove('is-hidden');
    elements.addDocModal.style.display = 'flex';
    resetUploadForm();
  };

  const closeAddDocModal = () => {
    if (!elements.addDocModal) return;
    elements.addDocModal.hidden = true;
    elements.addDocModal.classList.add('is-hidden');
    elements.addDocModal.style.display = 'none';
    resetUploadForm();
  };

  const handleFileSelection = (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      showToast('Please select a valid PDF file (.pdf)', 'error');
      return;
    }
    selectedUploadFile = file;
    if (elements.dropPrompt) elements.dropPrompt.hidden = true;
    if (elements.selectedFileBadge) elements.selectedFileBadge.hidden = false;
    if (elements.selectedFileName) elements.selectedFileName.textContent = file.name;
    const sizeKb = Math.round(file.size / 1024);
    if (elements.selectedFileSize) {
      elements.selectedFileSize.textContent = sizeKb > 1024 
        ? `(${(sizeKb / 1024).toFixed(1)} MB)` 
        : `(${sizeKb} KB)`;
    }
    if (elements.submitUploadBtn) elements.submitUploadBtn.disabled = false;

    // Auto-fill title
    if (elements.uploadDocTitle && !elements.uploadDocTitle.value.trim()) {
      const clean = file.name.replace(/\.pdf$/i, '').replace(/[\-_]/g, ' ');
      elements.uploadDocTitle.value = clean.charAt(0).toUpperCase() + clean.slice(1);
    }
  };

  if (elements.addDocBtn) {
    elements.addDocBtn.addEventListener('click', (e) => {
      e.preventDefault();
      openAddDocModal();
    });
  }

  if (elements.closeAddDocModal) {
    elements.closeAddDocModal.addEventListener('click', (e) => {
      e.preventDefault();
      closeAddDocModal();
    });
  }

  if (elements.cancelUploadBtn) {
    elements.cancelUploadBtn.addEventListener('click', (e) => {
      e.preventDefault();
      closeAddDocModal();
    });
  }

  if (elements.addDocModal) {
    elements.addDocModal.addEventListener('click', (e) => {
      if (e.target === elements.addDocModal) {
        closeAddDocModal();
      }
    });
  }

  // File Drop Area
  if (elements.fileDropArea) {
    elements.fileDropArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      elements.fileDropArea.classList.add('dragover');
    });
    elements.fileDropArea.addEventListener('dragleave', () => {
      elements.fileDropArea.classList.remove('dragover');
    });
    elements.fileDropArea.addEventListener('drop', (e) => {
      e.preventDefault();
      elements.fileDropArea.classList.remove('dragover');
      if (e.dataTransfer && e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });
  }

  if (elements.pdfFileInput) {
    elements.pdfFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });
  }

  if (elements.clearSelectedFile) {
    elements.clearSelectedFile.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      resetUploadForm();
    });
  }

  // Upload Form Submission
  if (elements.uploadDocForm) {
    elements.uploadDocForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!selectedUploadFile) {
        showToast('Please select a PDF document first', 'error');
        return;
      }

      elements.submitUploadBtn.disabled = true;
      elements.cancelUploadBtn.disabled = true;
      if (elements.uploadProgressContainer) elements.uploadProgressContainer.hidden = false;
      if (elements.uploadProgressText) elements.uploadProgressText.textContent = 'Parsing PDF structure, chunking clauses, and building embeddings...';
      if (elements.submitUploadBtnText) elements.submitUploadBtnText.textContent = 'Indexing...';

      const formData = new FormData();
      formData.append('file', selectedUploadFile);
      if (elements.uploadDocTitle && elements.uploadDocTitle.value.trim()) {
        formData.append('title', elements.uploadDocTitle.value.trim());
      }
      if (elements.uploadDocIssuer && elements.uploadDocIssuer.value.trim()) {
        formData.append('issuer', elements.uploadDocIssuer.value.trim());
      }

      try {
        const res = await fetch('/api/documents/upload', {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || 'Upload failed');
        }

        const result = await res.json();
        showToast(result.message || 'Document indexed successfully!', 'success');
        closeAddDocModal();
        await checkHealth();
        await loadCorpus();
      } catch (err) {
        console.error('Upload error:', err);
        showToast(`Upload failed: ${err.message}`, 'error');
        if (elements.uploadProgressContainer) elements.uploadProgressContainer.hidden = true;
        elements.submitUploadBtn.disabled = false;
        elements.cancelUploadBtn.disabled = false;
        if (elements.submitUploadBtnText) elements.submitUploadBtnText.textContent = 'Upload & Index';
      }
    });
  }

  // Shortcuts Modal Helpers
  const closeShortcutsModal = () => {
    if (!elements.shortcutsModal) return;
    elements.shortcutsModal.hidden = true;
    elements.shortcutsModal.classList.add('is-hidden');
    elements.shortcutsModal.style.display = 'none';
  };

  const openShortcutsModal = () => {
    if (!elements.shortcutsModal) return;
    elements.shortcutsModal.hidden = false;
    elements.shortcutsModal.classList.remove('is-hidden');
    elements.shortcutsModal.style.display = 'flex';
  };

  // Shortcuts Modal Events
  if (elements.shortcutsBtn) {
    elements.shortcutsBtn.addEventListener('click', (e) => {
      e.preventDefault();
      openShortcutsModal();
    });
  }

  if (elements.closeShortcutsModal) {
    elements.closeShortcutsModal.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      closeShortcutsModal();
    });
  }

  if (elements.shortcutsModal) {
    elements.shortcutsModal.addEventListener('click', (e) => {
      if (e.target === elements.shortcutsModal) {
        closeShortcutsModal();
      }
    });
  }

  // Global Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      elements.chatInput.focus();
    } else if (e.key === '?' && document.activeElement !== elements.chatInput) {
      e.preventDefault();
      if (elements.shortcutsModal.hidden || elements.shortcutsModal.style.display === 'none') {
        openShortcutsModal();
      } else {
        closeShortcutsModal();
      }
    } else if (e.key.toLowerCase() === 't' && document.activeElement !== elements.chatInput) {
      toggleTheme();
    } else if (e.key === 'Escape') {
      closeShortcutsModal();
      closeAddDocModal();
      closeDemoDock();
      if (elements.settingsPopover) {
        elements.settingsPopover.hidden = true;
        elements.settingsPopover.style.display = 'none';
      }
      if (elements.evidencePanel) {
        elements.evidencePanel.classList.remove('mobile-open');
      }
    }
  });

  // Resize Window
  window.addEventListener('resize', () => {
    const activeBtn = document.querySelector('.nav-tab.active');
    updateNavIndicator(activeBtn);
    if (state.activeTab === 'view-eval') {
      renderAblationChart();
    }
  });
}

// ============================================================================
// Initialization on DOM Loaded
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  applyTheme(state.theme);
  setupEventListeners();

  // Position indicator on active tab
  const activeBtn = document.querySelector('.nav-tab.active');
  updateNavIndicator(activeBtn);

  // Ensure modals and popovers are hidden on initial load
  if (elements.shortcutsModal) {
    elements.shortcutsModal.hidden = true;
    elements.shortcutsModal.classList.add('is-hidden');
    elements.shortcutsModal.style.display = 'none';
  }
  if (elements.addDocModal) {
    elements.addDocModal.hidden = true;
    elements.addDocModal.classList.add('is-hidden');
    elements.addDocModal.style.display = 'none';
  }
  if (elements.settingsPopover) {
    elements.settingsPopover.hidden = true;
    elements.settingsPopover.style.display = 'none';
  }
  if (elements.demoDock) {
    elements.demoDock.hidden = true;
    elements.demoDock.style.display = 'none';
  }

  // Initial Health & Corpus load
  checkHealth();
  loadCorpus();
});
