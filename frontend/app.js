// ─── State ──────────────────────────────────────────────
const STATE = {
    hasDataset: false,
    data: null,
    theme: localStorage.getItem('datalens-theme') || 'dark'
};

// ─── DOM References ──────────────────────────────────────
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

const el = {
    html: document.documentElement,
    themeToggle: $('theme-toggle'),
    themeIcon: $('theme-icon'),
    fileInput: $('dataset-file'),
    sidebarUploadBtn: $('sidebar-upload-btn'),
    loadingOverlay: $('loading-overlay'),
    emptyState: $('empty-state'),
    dropZone: $('drop-zone'),
    fileIndicatorName: $('file-indicator-name'),
    pageTitle: $('page-title'),
    loadingSteps: $$('.loading-step'),
    navLinks: $$('.nav-link'),
    views: {
        overview: $('view-overview'),
        structure: $('view-structure'),
        insights: $('view-insights'),
        visualizations: $('view-visualizations'),
        chatbot: $('view-chatbot')
    },
    // Overview
    statRows: $('stat-rows'),
    statCols: $('stat-cols'),
    statMissing: $('stat-missing'),
    statDupes: $('stat-dupes'),
    statSizeLabel: $('stat-size-label'),
    statColBreakdown: $('stat-col-breakdown'),
    statMissingPct: $('stat-missing-pct'),
    statMemory: $('stat-memory'),
    overviewSummary: $('overview-summary'),
    detectBadge: $('detect-badge'),
    detectTarget: $('detect-target'),
    detectUniques: $('detect-uniques'),
    detectImbalanced: $('detect-imbalanced'),
    detectReason: $('detect-reason'),
    // Structure
    colTypeChips: $('col-type-chips'),
    colCountBadge: $('col-count-badge'),
    colTableBody: $('col-table-body'),
    // Insights
    preprocessingList: $('preprocessing-list'),
    riskList: $('risk-list'),
    modelGrid: $('model-grid'),
    // Visualizations
    imgMissing: $('img-missing'),
    emptyMissing: $('empty-missing'),
    imgTarget: $('img-target'),
    emptyTarget: $('empty-target'),
    imgCorr: $('img-corr'),
    emptyCorr: $('empty-corr'),
    // Chat
    chatMessages: $('chat-messages'),
    chatInput: $('chat-input'),
    chatSend: $('chat-send'),
    quickChips: $$('.chip')
};

// ─── THEME ───────────────────────────────────────────────
function applyTheme(theme) {
    STATE.theme = theme;
    el.html.setAttribute('data-theme', theme);
    localStorage.setItem('datalens-theme', theme);
    el.themeIcon.className = theme === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
    el.themeToggle.title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
}

el.themeToggle.addEventListener('click', () => {
    applyTheme(STATE.theme === 'dark' ? 'light' : 'dark');
});

// Apply saved theme immediately
applyTheme(STATE.theme);

// ─── NAVIGATION ──────────────────────────────────────────
const PAGE_TITLES = {
    overview:       'Overview',
    structure:      'Dataset Structure',
    insights:       'ML Insights',
    visualizations: 'Visualizations',
    chatbot:        'AI Chatbot'
};

function navigateTo(sectionId) {
    if (!STATE.hasDataset && sectionId !== 'overview') return;

    // Update nav links
    el.navLinks.forEach(link => {
        const isTarget = link.dataset.section === sectionId;
        link.classList.toggle('active', isTarget);
    });

    // Show/hide views
    Object.entries(el.views).forEach(([key, view]) => {
        view.classList.toggle('hidden', key !== sectionId);
    });

    el.pageTitle.textContent = PAGE_TITLES[sectionId] || 'DataLens';

    if (sectionId === 'chatbot') scrollChatToBottom();
}

el.navLinks.forEach(link => {
    link.addEventListener('click', e => {
        e.preventDefault();
        navigateTo(link.dataset.section);
    });
});

// ─── FILE UPLOAD ─────────────────────────────────────────
async function handleFileUpload(file) {
    if (!file) return;

    const validExtensions = ['.csv', '.xlsx', '.xls'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExtensions.includes(ext)) {
        alert('Unsupported file type. Please upload a CSV, XLSX or XLS file.');
        return;
    }

    // Show loading screen and animate steps
    el.loadingOverlay.classList.remove('hidden');
    el.emptyState.classList.add('hidden');
    animateLoadingSteps();

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', { method: 'POST', body: formData });

        if (!response.ok) {
            let detail = 'Upload failed';
            try { detail = (await response.json()).detail || detail; } catch {}
            throw new Error(detail);
        }

        const data = await response.json();
        STATE.hasDataset = true;
        STATE.data = data;

        populateDashboard(data, file.name);

    } catch (err) {
        el.emptyState.classList.remove('hidden');
        alert(`Error: ${err.message}`);
        console.error(err);
    } finally {
        el.loadingOverlay.classList.add('hidden');
        if (loadingInterval) clearInterval(loadingInterval);
    }
}

let loadingInterval = null;

function animateLoadingSteps() {
    let stepIdx = 0;
    el.loadingSteps.forEach(s => s.classList.remove('active'));
    el.loadingSteps[0].classList.add('active');
    loadingInterval = setInterval(() => {
        stepIdx = (stepIdx + 1) % el.loadingSteps.length;
        el.loadingSteps.forEach(s => s.classList.remove('active'));
        el.loadingSteps[stepIdx].classList.add('active');
    }, 900);
}

el.fileInput.addEventListener('change', e => handleFileUpload(e.target.files[0]));
el.sidebarUploadBtn.addEventListener('click', () => el.fileInput.click());

// Drag-and-drop on the drop zone
el.dropZone?.addEventListener('dragover', e => {
    e.preventDefault();
    el.dropZone.classList.add('dragover');
});
el.dropZone?.addEventListener('dragleave', () => el.dropZone.classList.remove('dragover'));
el.dropZone?.addEventListener('drop', e => {
    e.preventDefault();
    el.dropZone.classList.remove('dragover');
    handleFileUpload(e.dataTransfer.files[0]);
});

// ─── POPULATE DASHBOARD ───────────────────────────────────
function populateDashboard(data, fileName) {
    const { basic_info, column_types, dataset_profile, task_info, target_analysis,
            summary_text, preprocessing_steps, detected_risks, models, reasons,
            data_types, null_info, charts } = data;

    // Update sidebar file indicator
    el.fileIndicatorName.textContent = fileName;

    // Enable all nav links
    el.navLinks.forEach(l => l.classList.remove('disabled'));

    // ── Overview ──
    el.statRows.textContent = fmt(basic_info.rows);
    el.statSizeLabel.textContent = `${dataset_profile.size_label} dataset`;

    el.statCols.textContent = fmt(basic_info.columns);
    el.statColBreakdown.textContent = `${column_types.numerical.length} num · ${column_types.categorical.length} cat`;

    el.statMissing.textContent = fmt(dataset_profile.total_missing);
    el.statMissingPct.textContent = `${dataset_profile.missing_percentage}% of all cells`;

    el.statDupes.textContent = fmt(basic_info.duplicate_rows);
    el.statMemory.textContent = `${basic_info.memory_usage_kb} KB memory`;

    el.overviewSummary.textContent = summary_text;

    el.detectBadge.textContent = task_info.problem_type;
    el.detectTarget.textContent = task_info.possible_target || 'None';
    el.detectUniques.textContent = target_analysis.unique_values !== null ? target_analysis.unique_values : 'N/A';
    el.detectImbalanced.textContent = target_analysis.target_found
        ? (target_analysis.is_imbalanced ? 'Yes ⚠' : 'No ✓') : 'N/A';
    el.detectReason.textContent = task_info.reason || '';

    // ── Structure ──
    el.colTypeChips.innerHTML = buildTypeChips(column_types);
    el.colCountBadge.textContent = `${basic_info.columns} columns`;

    // Build null lookup
    const nullMap = {};
    null_info.forEach(row => { nullMap[row.Column] = row; });

    el.colTableBody.innerHTML = '';
    data_types.forEach((row, i) => {
        const nullData = nullMap[row.Column] || { 'Null Count': 0, 'Null Percentage': 0 };
        const nullCount = nullData['Null Count'] || 0;
        const nullPct = nullData['Null Percentage'] || 0;
        const typeTag = getTypeTag(row.Column, column_types);
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="text-muted" style="font-family:var(--font-mono);font-size:11px;">${i + 1}</td>
            <td class="col-name">${row.Column}</td>
            <td>${typeTag}</td>
            <td style="font-family:var(--font-mono);font-size:11px;">${row['Data Type']}</td>
            <td class="${nullCount > 0 ? 'has-missing' : ''}">${fmt(nullCount)}</td>
            <td class="${nullCount > 0 ? 'has-missing' : ''}">${nullPct}%</td>
        `;
        el.colTableBody.appendChild(tr);
    });

    // ── ML Insights ──
    el.preprocessingList.innerHTML = '';
    preprocessing_steps.forEach(step => {
        const li = document.createElement('li');
        li.textContent = step;
        el.preprocessingList.appendChild(li);
    });

    el.riskList.innerHTML = '';
    if (!detected_risks.length) {
        const li = document.createElement('li');
        li.className = 'risk-ok';
        li.textContent = 'No critical data risks detected. Dataset looks structurally sound.';
        el.riskList.appendChild(li);
    } else {
        detected_risks.forEach(risk => {
            const li = document.createElement('li');
            li.textContent = risk;
            el.riskList.appendChild(li);
        });
    }

    el.modelGrid.innerHTML = '';
    models.forEach((name, idx) => {
        const div = document.createElement('div');
        div.className = 'model-chip';
        div.innerHTML = `
            <div class="model-chip-name"><i class="fa-solid fa-bolt"></i>${name}</div>
            <div class="model-chip-reason">${reasons[idx]}</div>
        `;
        el.modelGrid.appendChild(div);
    });

    // ── Visualizations ──
    renderChart(charts.missing_values, el.imgMissing, el.emptyMissing);
    renderChart(charts.target_distribution, el.imgTarget, el.emptyTarget);
    renderChart(charts.correlation_heatmap, el.imgCorr, el.emptyCorr);

    // ── Reset Chatbot ──
    el.chatMessages.innerHTML = '';
    appendBotMessage(`Hi! I've finished analyzing **${fileName}**.\n\nThis looks like a **${task_info.problem_type}** problem. The dataset has ${fmt(basic_info.rows)} rows and ${fmt(basic_info.columns)} columns.\n\nAsk me anything — try the quick chips below or type your own question.`);

    navigateTo('overview');
}

// ─── HELPERS ─────────────────────────────────────────────
function fmt(n) {
    return Number(n).toLocaleString();
}

function renderChart(b64, imgEl, emptyEl) {
    if (b64) {
        imgEl.src = `data:image/png;base64,${b64}`;
        imgEl.classList.remove('hidden');
        emptyEl.classList.add('hidden');
    } else {
        imgEl.classList.add('hidden');
        emptyEl.classList.remove('hidden');
    }
}

function getTypeTag(col, column_types) {
    if (column_types.numerical.includes(col))   return '<span class="type-tag tt-num">numerical</span>';
    if (column_types.categorical.includes(col)) return '<span class="type-tag tt-cat">categorical</span>';
    if (column_types.datetime.includes(col))    return '<span class="type-tag tt-dt">datetime</span>';
    if (column_types.boolean.includes(col))     return '<span class="type-tag tt-bool">boolean</span>';
    if (column_types.id_like.includes(col))     return '<span class="type-tag tt-id">id/key</span>';
    if (column_types.text.includes(col))        return '<span class="type-tag tt-text">text</span>';
    return '<span class="type-tag tt-?">unknown</span>';
}

function buildTypeChips(column_types) {
    const types = [
        { key: 'numerical',   label: 'Numerical',   cls: 'tc-num' },
        { key: 'categorical', label: 'Categorical', cls: 'tc-cat' },
        { key: 'datetime',    label: 'Datetime',    cls: 'tc-dt'  },
        { key: 'boolean',     label: 'Boolean',     cls: 'tc-bool'},
        { key: 'id_like',     label: 'ID / Keys',   cls: 'tc-id'  },
        { key: 'text',        label: 'Text',         cls: 'tc-text'},
    ];
    return types.map(t => `
        <div class="type-chip ${t.cls}">
            ${t.label}
            <span class="chip-count">${column_types[t.key].length}</span>
        </div>
    `).join('');
}

// ─── CHATBOT ─────────────────────────────────────────────
function appendBotMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'msg bot-msg';
    // simple bold **text** support
    const formatted = text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    msg.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="msg-bubble"><p>${formatted}</p></div>
    `;
    el.chatMessages.appendChild(msg);
    scrollChatToBottom();
    return msg;
}

function appendUserMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'msg user-msg';
    msg.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-user"></i></div>
        <div class="msg-bubble"><p>${text}</p></div>
    `;
    el.chatMessages.appendChild(msg);
    scrollChatToBottom();
}

function appendTypingIndicator() {
    const msg = document.createElement('div');
    msg.className = 'msg bot-msg';
    msg.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="msg-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>
    `;
    el.chatMessages.appendChild(msg);
    scrollChatToBottom();
    return msg;
}

function scrollChatToBottom() {
    el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
}

async function sendChatQuery(query) {
    if (!query.trim()) return;
    if (!STATE.hasDataset) { alert('Please upload a dataset first!'); return; }

    el.chatInput.value = '';
    appendUserMessage(query);
    const typingEl = appendTypingIndicator();

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        typingEl.remove();

        if (!res.ok) {
            let detail = 'Chat request failed';
            try { detail = (await res.json()).detail || detail; } catch {}
            appendBotMessage(`⚠️ ${detail}`);
            return;
        }

        const data = await res.json();
        appendBotMessage(data.response);

    } catch (err) {
        typingEl.remove();
        appendBotMessage(`⚠️ Could not reach the server: ${err.message}`);
    }
}

el.chatSend.addEventListener('click', () => sendChatQuery(el.chatInput.value));
el.chatInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendChatQuery(el.chatInput.value); });

el.quickChips.forEach(chip => {
    chip.addEventListener('click', () => sendChatQuery(chip.dataset.q));
});
