/* ===== Campus Knowledge QA System - Frontend Application ===== */

const API_BASE = '/api';

/* ---------- State ---------- */
const state = {
    currentUser: null,       // { id, name, student_id, phone }
    isAdmin: false,
    currentSessionId: null,
    sessions: [],            // [{ session_id, title, active }]
    messages: [],            // [{ role, content, query?, answer? }]
    availableModels: [],     // [{ id, model_id, name, api_base, is_default }]
    selectedModelId: null,   // numeric DB id of the selected model
    isGenerating: false,     // whether AI is currently generating
    currentAbortController: null, // AbortController for stopping generation
};

/* ---------- Helpers ---------- */
function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function showPage(pageId) {
    $$('.page').forEach(p => p.classList.remove('active'));
    const el = document.getElementById(pageId);
    if (el) el.classList.add('active');
}

function showToast(msg, duration = 2500) {
    const t = $('#toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), duration);
}

function showConfirmModal(title, message, confirmText, onConfirm) {
    const overlay = document.getElementById('confirm-modal-overlay');
    document.getElementById('confirm-modal-title').textContent = title;
    document.getElementById('confirm-modal-body').textContent = message;
    const confirmBtn = document.getElementById('confirm-modal-confirm');
    confirmBtn.textContent = confirmText || '删除';
    overlay.style.display = '';

    const close = () => { overlay.style.display = 'none'; };
    const handleConfirm = () => { close(); onConfirm(); };
    const handleCancel = () => close();
    const handleOverlay = (e) => { if (e.target === overlay) close(); };

    confirmBtn.onclick = handleConfirm;
    document.getElementById('confirm-modal-cancel').onclick = handleCancel;
    overlay.onclick = handleOverlay;
}

async function apiFetch(path, options = {}) {
    const url = API_BASE + path;
    const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
    });
    const data = await res.json();
    if (!res.ok) {
        throw new Error(data.detail || `HTTP ${res.status}`);
    }
    return data;
}

/* ---------- Navigation Helpers ---------- */
document.addEventListener('click', e => {
    const link = e.target.closest('[data-goto]');
    if (link) {
        e.preventDefault();
        showPage(link.dataset.goto);
    }
    // Close user menu popup when clicking outside
    const userMenuPopup = document.getElementById('user-menu-popup');
    if (userMenuPopup && !e.target.closest('#btn-user-menu') && !e.target.closest('#user-menu-popup')) {
        userMenuPopup.classList.remove('open');
    }
});

/* ========== AUTH: Login ========== */
$('#login-form').addEventListener('submit', async e => {
    e.preventDefault();
    const errEl = $('#login-error');
    errEl.textContent = '';
    const account = $('#login-account').value.trim();
    const password = $('#login-password').value;
    if (!account || !password) { errEl.textContent = '请填写完整信息'; return; }
    try {
        const data = await apiFetch('/auth/login', {
            method: 'POST',
            body: JSON.stringify({ account, password }),
        });
        state.currentUser = data.user;
        state.isAdmin = false;
        enterChat();
    } catch (err) {
        if (err.message && err.message.includes('invalid account or password')) {
            errEl.textContent = '账号未注册或密码错误，请检查后重试';
        } else {
            errEl.textContent = err.message;
        }
    }
});

/* ========== AUTH: Register ========== */
$('#register-form').addEventListener('submit', async e => {
    e.preventDefault();
    const errEl = $('#register-error');
    errEl.textContent = '';
    const name = $('#reg-name').value.trim();
    const student_id = $('#reg-sid').value.trim();
    const phone = $('#reg-phone').value.trim();
    const password = $('#reg-password').value;
    if (!name || !student_id || !phone || !password) { errEl.textContent = '请填写完整信息'; return; }
    try {
        const data = await apiFetch('/auth/register', {
            method: 'POST',
            body: JSON.stringify({ name, student_id, phone, password }),
        });
        showToast('注册成功，请登录');
        showPage('page-login');
    } catch (err) {
        errEl.textContent = err.message;
    }
});

/* ========== AUTH: Admin Login ========== */
$('#admin-login-form').addEventListener('submit', async e => {
    e.preventDefault();
    const errEl = $('#admin-login-error');
    errEl.textContent = '';
    const username = $('#admin-username').value.trim();
    const password = $('#admin-password').value;
    if (!username || !password) { errEl.textContent = '请填写完整信息'; return; }
    try {
        await apiFetch('/auth/admin/login', {
            method: 'POST',
            body: JSON.stringify({ username, password }),
        });
        state.isAdmin = true;
        enterAdmin();
    } catch (err) {
        errEl.textContent = err.message;
    }
});

/* ========== Chat Page ========== */
function enterChat() {
    showPage('page-chat');
    const userName = state.currentUser?.name || '同学';
    $('#user-display-name').textContent = userName;
    $('#user-display-id').textContent = state.currentUser?.student_id || '';
    $('#greet-title').textContent = `Hi，${userName}同学，我是你的校园问答助理！`;
    // Persist login state for page refresh
    sessionStorage.setItem('campusqa_user', JSON.stringify(state.currentUser));
    sessionStorage.setItem('campusqa_page', 'page-chat');
    resetToWelcome();
    loadSessions();
    loadChatModels();
    triggerWarmup();
}

/* ----- Theme Toggle ----- */
function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('campusqa_theme', theme);
    const label = document.getElementById('theme-label');
    if (label) label.textContent = theme === 'dark' ? '浅色模式' : '深色模式';
}

(function initTheme() {
    const saved = localStorage.getItem('campusqa_theme') || 'light';
    applyTheme(saved);
})();

document.getElementById('btn-toggle-theme').addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    applyTheme(current === 'dark' ? 'light' : 'dark');
    document.getElementById('user-menu-popup').classList.remove('open');
});

/* ----- User Menu (three-dot) & Logout ----- */
$('#btn-user-menu').addEventListener('click', e => {
    e.stopPropagation();
    $('#user-menu-popup').classList.toggle('open');
});
$('#btn-chat-logout').addEventListener('click', () => {
    $('#user-menu-popup').classList.remove('open');
    state.currentUser = null;
    state.currentSessionId = null;
    state.sessions = [];
    state.messages = [];
    sessionStorage.removeItem('campusqa_user');
    sessionStorage.removeItem('campusqa_page');
    showPage('page-login');
});

/* ----- 冷启动预热：登录后自动触发，消除首次提问延迟 ----- */
function triggerWarmup() {
    fetch(API_BASE + '/warmup', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            console.log('[Warmup] 预热已触发:', data.status);
        })
        .catch(err => {
            console.warn('[Warmup] 预热请求失败（不影响使用）:', err.message);
        });
}

async function loadSessions() {
    try {
        const uid = state.currentUser?.id;
        const url = uid ? `/sessions?user_id=${uid}` : '/sessions';
        const data = await apiFetch(url);
        state.sessions = (data.items || []).map(s => ({
            session_id: s.session_id,
            title: s.title,
        }));
        renderHistoryList();
    } catch (err) {
        // silently fail - sessions list is not critical
    }
}

/* ----- Chat Model Selector ----- */
async function loadChatModels() {
    try {
        const data = await apiFetch('/models');
        state.availableModels = (data.models || []).filter(m => m.enabled !== false);
        const def = state.availableModels.find(m => m.is_default);
        state.selectedModelId = def ? def.id : (state.availableModels[0]?.id || null);
        renderModelSelector();
    } catch (err) {
        const el = document.getElementById('selected-model-name');
        if (el) el.textContent = '默认模型';
    }
}

function _buildModelPopup(popupEl, onSelect) {
    popupEl.innerHTML = '';
    if (!state.availableModels.length) {
        const empty = document.createElement('div');
        empty.className = 'model-popup-item';
        empty.textContent = '暂无可用模型';
        popupEl.appendChild(empty);
        return;
    }
    state.availableModels.forEach(m => {
        const item = document.createElement('div');
        item.className = 'model-popup-item' + (m.id === state.selectedModelId ? ' active' : '');
        item.innerHTML = `<span>${escapeHtml(m.name)}</span>`;
        item.addEventListener('click', e => {
            e.stopPropagation();
            state.selectedModelId = m.id;
            renderModelSelector();
            popupEl.classList.remove('open');
            onSelect && onSelect();
        });
        popupEl.appendChild(item);
    });
}

function renderModelSelector() {
    const selected = state.availableModels.find(m => m.id === state.selectedModelId);
    const label = selected ? selected.name : '选择模型';
    const nameEl = document.getElementById('selected-model-name');
    if (nameEl) nameEl.textContent = label;
    const convNameEl = document.getElementById('conv-model-name');
    if (convNameEl) convNameEl.textContent = label;
    const dd = document.getElementById('model-dropdown');
    if (dd) _buildModelPopup(dd, null);
    const cdd = document.getElementById('conv-model-dropdown');
    if (cdd) _buildModelPopup(cdd, null);
}

$('#model-selector-trigger').addEventListener('click', e => {
    e.stopPropagation();
    const dd = $('#model-dropdown');
    dd.classList.toggle('open');
    const cdd = document.getElementById('conv-model-dropdown');
    if (cdd) cdd.classList.remove('open');
});
document.getElementById('conv-model-trigger').addEventListener('click', e => {
    e.stopPropagation();
    const cdd = $('#conv-model-dropdown');
    cdd.classList.toggle('open');
    const dd = document.getElementById('model-dropdown');
    if (dd) dd.classList.remove('open');
});
document.addEventListener('click', () => {
    const dd = document.getElementById('model-dropdown');
    if (dd) dd.classList.remove('open');
    const cdd = document.getElementById('conv-model-dropdown');
    if (cdd) cdd.classList.remove('open');
});

function resetToWelcome() {
    state.currentSessionId = null;
    state.messages = [];
    $('#welcome-section').style.display = '';
    $('#conversation-section').style.display = 'none';
    $('#welcome-input').value = '';
    $('#messages-area').innerHTML = '';
    highlightActiveSession(null);
}

function enterConversation(sessionId, title) {
    state.currentSessionId = sessionId;
    if (!sessionId) {
        $('#messages-area').innerHTML = '';
    }
    $('#welcome-section').style.display = 'none';
    $('#conversation-section').style.display = '';
    $('#conv-title').textContent = title || '新对话';
    $('#conv-input').value = '';
    $('#conv-input').focus();
}

function highlightActiveSession(sessionId) {
    $$('.history-item').forEach(el => {
        el.classList.toggle('active', el.dataset.session === sessionId);
    });
}

/* ----- Send Message (Streaming) ----- */
async function sendMessage(query) {
    if (!query.trim()) return;
    if (state.isGenerating) return;

    const isNew = !state.currentSessionId;
    if (isNew) {
        enterConversation(null, query.slice(0, 20));
    }

    appendUserMessage(query);
    await _doStreamingChat(query, isNew);
}

async function _doStreamingChat(query, isNew) {
    state.isGenerating = true;
    const abortController = new AbortController();
    state.currentAbortController = abortController;

    // 创建流式AI消息容器
    const { msgDiv, contentEl, msgId } = appendAIStreaming();
    showStopButton(true);

    let fullAnswer = '';
    let receivedSessionId = null;

    try {
        const res = await fetch(API_BASE + '/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                session_id: state.currentSessionId || undefined,
                model_id: state.selectedModelId || undefined,
                user_id: state.currentUser?.id || undefined,
            }),
            signal: abortController.signal,
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP ${res.status}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        let sseEvent = '';
        let sseDataLines = [];

        function dispatchSSE() {
            if (!sseEvent && sseDataLines.length === 0) return;
            const data = sseDataLines.join('\n');
            if (sseEvent === 'session') {
                try {
                    const parsed = JSON.parse(data);
                    receivedSessionId = parsed.session_id;
                    state.currentSessionId = receivedSessionId;
                } catch (e) { /* ignore */ }
            } else if (sseEvent === 'chunk') {
                fullAnswer += data;
                contentEl.innerHTML = renderMarkdown(fullAnswer);
                const wrapper = $('#messages-area').closest('.messages-scroll-wrapper');
                if (wrapper) wrapper.scrollTop = wrapper.scrollHeight;
            } else if (sseEvent === 'done') {
                try {
                    const parsed = JSON.parse(data);
                    receivedSessionId = parsed.session_id;
                } catch (e) { /* ignore */ }
            } else if (sseEvent === 'error') {
                fullAnswer += data;
                contentEl.innerHTML = renderMarkdown(fullAnswer);
            }
            sseEvent = '';
            sseDataLines = [];
        }

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const rawLine of lines) {
                const line = rawLine.replace(/\r$/, '');
                if (line === '') {
                    dispatchSSE();
                } else if (line.startsWith('event: ')) {
                    sseEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ')) {
                    sseDataLines.push(line.slice(6));
                } else if (line === 'data:') {
                    sseDataLines.push('');
                }
            }
        }
        dispatchSSE();
    } catch (err) {
        if (err.name === 'AbortError') {
            // 用户点击了停止生成
            if (!fullAnswer) fullAnswer = '（已停止生成）';
        } else {
            fullAnswer = '抱歉，发生了错误：' + err.message;
            contentEl.innerHTML = renderMarkdown(fullAnswer);
        }
    }

    // 流式完成，替换为完整消息（带操作按钮）
    finalizeStreamingMessage(msgDiv, query, fullAnswer, msgId);
    showStopButton(false);
    state.isGenerating = false;
    state.currentAbortController = null;

    if (receivedSessionId) {
        state.currentSessionId = receivedSessionId;
        if (isNew) {
            addSessionToHistory(receivedSessionId, query.slice(0, 20));
        }
        highlightActiveSession(receivedSessionId);
        $('#conv-title').textContent = query.slice(0, 20);
    }
}

function stopGeneration() {
    if (state.currentAbortController) {
        state.currentAbortController.abort();
    }
}

async function regenerateMessage(query) {
    if (state.isGenerating) return;
    // 删除最后一条AI消息，不删除用户消息
    const area = $('#messages-area');
    const lastAI = area.querySelector('.msg-ai:last-child');
    if (lastAI) lastAI.remove();
    // 重新生成（不重新追加用户消息）
    await _doStreamingChat(query, false);
}

function appendAIStreaming() {
    const area = $('#messages-area');
    const div = document.createElement('div');
    div.className = 'msg-ai';
    const msgId = 'msg-' + Date.now();
    div.innerHTML = `
        <div class="ai-column">
            <div class="ai-bubble">
                <div class="ai-detail" id="${msgId}"><div class="loading-dots"><span></span><span></span><span></span></div></div>
            </div>
        </div>
    `;
    area.appendChild(div);
    const w1 = area.closest('.messages-scroll-wrapper');
    if (w1) w1.scrollTop = w1.scrollHeight;
    return { msgDiv: div, contentEl: div.querySelector('.ai-detail'), msgId };
}

function finalizeStreamingMessage(msgDiv, query, answer, msgId) {
    const column = msgDiv.querySelector('.ai-column');
    // 更新内容
    const detail = column.querySelector('.ai-detail');
    detail.innerHTML = renderMarkdown(answer);
    detail.id = msgId;

    // 添加操作按钮（复制 + 反馈 + 重新生成）
    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'action-buttons';
    const _fbId1 = ++_feedbackStoreSeq;
    _feedbackStore.set(_fbId1, { query, answer });
    actionsDiv.innerHTML = `
        <button class="action-btn" data-tip="复制" onclick="copyText(this, '${msgId}')"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>
        <button class="action-btn" data-tip="反馈" onclick="openFeedback(${_fbId1})"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 14V2"/><path d="M9 18.12L2 22V8l7-4z"/><path d="M23 8l-7 4v10l7-4z"/></svg></button>
        <button class="action-btn action-btn-regen" data-tip="重新生成" onclick="regenerateMessage('${escapeAttr(query)}')"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg></button>
    `;
    column.appendChild(actionsDiv);

    // 添加footer
    const footer = document.createElement('div');
    footer.className = 'ai-footer';
    footer.textContent = '内容由AI生成，仅供参考';
    column.appendChild(footer);
}

const _SEND_ICON = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>`;
const _STOP_ICON = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>`;

function showStopButton(show) {
    const btn = document.getElementById('conv-send');
    if (!btn) return;
    if (show) {
        btn.innerHTML = _STOP_ICON;
        btn.classList.add('btn-send-round--stop');
        btn._isStop = true;
    } else {
        btn.innerHTML = _SEND_ICON;
        btn.classList.remove('btn-send-round--stop');
        btn._isStop = false;
    }
}

function appendUserMessage(text) {
    const area = $('#messages-area');
    const div = document.createElement('div');
    div.className = 'msg-user';
    div.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
    area.appendChild(div);
    const w2 = area.closest('.messages-scroll-wrapper');
    if (w2) w2.scrollTop = w2.scrollHeight;
}

function appendAILoading() {
    const area = $('#messages-area');
    const div = document.createElement('div');
    div.className = 'msg-ai';
    div.innerHTML = `<div class="ai-column"><div class="ai-bubble"><div class="loading-dots"><span></span><span></span><span></span></div></div></div>`;
    area.appendChild(div);
    const w3 = area.closest('.messages-scroll-wrapper');
    if (w3) w3.scrollTop = w3.scrollHeight;
    return div;
}

function appendAIMessage(query, answer) {
    const area = $('#messages-area');
    const div = document.createElement('div');
    div.className = 'msg-ai';
    const msgId = 'msg-' + Date.now();
    const _fbId2 = ++_feedbackStoreSeq;
    _feedbackStore.set(_fbId2, { query, answer });
    div.innerHTML = `
        <div class="ai-column">
            <div class="ai-bubble">
                <div class="ai-detail">${renderMarkdown(answer)}</div>
            </div>
            <div class="action-buttons">
                <button class="action-btn" data-tip="复制" onclick="copyText(this, '${msgId}')"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>
                <button class="action-btn" data-tip="反馈" onclick="openFeedback(${_fbId2})"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 14V2"/><path d="M9 18.12L2 22V8l7-4z"/><path d="M23 8l-7 4v10l7-4z"/></svg></button>
                <button class="action-btn action-btn-regen" data-tip="重新生成" onclick="regenerateMessage('${escapeAttr(query)}')"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg></button>
            </div>
            <div class="ai-footer">内容由AI生成，仅供参考</div>
        </div>
    `;
    div.querySelector('.ai-detail').id = msgId;
    area.appendChild(div);
    const w4 = area.closest('.messages-scroll-wrapper');
    if (w4) w4.scrollTop = w4.scrollHeight;
}

function removeElement(el) { if (el && el.parentNode) el.parentNode.removeChild(el); }

function escapeHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/\n/g,'<br>');
}

function renderMarkdown(raw) {
    if (!raw) return '';
    // 轻量级内置 Markdown 渲染器，无外部依赖
    const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

    // 1. 提取代码块，避免内部被二次处理
    const codeBlocks = [];
    let text = raw.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        const idx = codeBlocks.length;
        codeBlocks.push(`<pre><code>${esc(code.replace(/\n$/, ''))}</code></pre>`);
        return `\x00CB${idx}\x00`;
    });

    // 2. 转义 HTML 特殊字符
    text = esc(text);

    // 3. 行内样式
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

    // 4. 按行处理块级元素
    const lines = text.split('\n');
    let html = '';
    let inUl = false, inOl = false;

    for (let i = 0; i < lines.length; i++) {
        let line = lines[i];

        // 标题
        const hMatch = line.match(/^(#{1,4})\s+(.+)$/);
        if (hMatch) {
            if (inUl) { html += '</ul>'; inUl = false; }
            if (inOl) { html += '</ol>'; inOl = false; }
            const level = hMatch[1].length;
            html += `<h${level}>${hMatch[2]}</h${level}>`;
            continue;
        }

        // 无序列表
        if (/^[\-\*]\s+/.test(line)) {
            if (inOl) { html += '</ol>'; inOl = false; }
            if (!inUl) { html += '<ul>'; inUl = true; }
            html += `<li>${line.replace(/^[\-\*]\s+/, '')}</li>`;
            continue;
        }

        // 有序列表
        if (/^\d+\.\s+/.test(line)) {
            if (inUl) { html += '</ul>'; inUl = false; }
            if (!inOl) { html += '<ol>'; inOl = true; }
            html += `<li>${line.replace(/^\d+\.\s+/, '')}</li>`;
            continue;
        }

        // 关闭列表
        if (inUl) { html += '</ul>'; inUl = false; }
        if (inOl) { html += '</ol>'; inOl = false; }

        // 分隔线
        if (/^---+$/.test(line.trim())) { html += '<hr>'; continue; }

        // 空行 → 段落间距
        if (line.trim() === '') { html += '<br>'; continue; }

        // 普通文本段落
        html += `<p>${line}</p>`;
    }
    if (inUl) html += '</ul>';
    if (inOl) html += '</ol>';

    // 5. 还原代码块
    html = html.replace(/\x00CB(\d+)\x00/g, (_, idx) => codeBlocks[+idx]);

    return html;
}
function escapeAttr(str) {
    return str.replace(/\\/g,'\\\\').replace(/'/g,"\\'").replace(/\n/g,'\\n');
}

function copyText(btn, msgId) {
    const el = document.getElementById(msgId);
    if (el) {
        navigator.clipboard.writeText(el.innerText).then(() => showToast('已复制到剪贴板'));
    }
}

/* ----- Feedback ----- */
const _feedbackStore = new Map();
let _feedbackStoreSeq = 0;

function openFeedback(storeId) {
    const { query, answer } = _feedbackStore.get(storeId) || {};
    _feedbackStore.delete(storeId);
    const modal = document.createElement('div');
    modal.className = 'feedback-modal';
    modal.innerHTML = `
        <div class="feedback-card">
            <h3>提交反馈</h3>
            <textarea id="feedback-text" placeholder="请输入您的反馈意见..."></textarea>
            <div class="feedback-actions">
                <button class="btn-cancel" onclick="this.closest('.feedback-modal').remove()">取消</button>
                <button class="btn-submit-feedback" id="btn-do-feedback">提交</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    modal.querySelector('#btn-do-feedback').addEventListener('click', async () => {
        const text = modal.querySelector('#feedback-text').value.trim();
        if (!text) { showToast('请输入反馈内容'); return; }
        try {
            await apiFetch('/feedback', {
                method: 'POST',
                body: JSON.stringify({
                    feedback_text: text,
                    user_id: state.currentUser?.id || null,
                    user_name: state.currentUser?.name || '',
                    query_content: query,
                    answer_content: answer,
                }),
            });
            showToast('反馈已提交，感谢！');
            modal.remove();
        } catch (err) {
            showToast('提交失败：' + err.message);
        }
    });
}

/* ----- Session Rename ----- */
async function renameSession(sessionId, newTitle) {
    try {
        await apiFetch(`/sessions/${sessionId}/title`, {
            method: 'PATCH',
            body: JSON.stringify({ title: newTitle }),
        });
        const s = state.sessions.find(s => s.session_id === sessionId);
        if (s) s.title = newTitle;
        renderHistoryList();
    } catch (err) {
        showToast('重命名失败：' + err.message);
    }
}

function initConvTitleEdit() {
    const span = $('#conv-title');
    span.addEventListener('click', () => {
        if (!state.currentSessionId) return;
        const currentTitle = span.textContent;
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'top-title-input';
        input.value = currentTitle;
        span.replaceWith(input);
        input.focus();
        input.select();

        const commit = async () => {
            const newTitle = input.value.trim();
            const replacement = document.createElement('span');
            replacement.className = 'top-title';
            replacement.id = 'conv-title';
            replacement.textContent = newTitle || currentTitle;
            input.replaceWith(replacement);
            initConvTitleEdit();
            if (newTitle && newTitle !== currentTitle && state.currentSessionId) {
                await renameSession(state.currentSessionId, newTitle);
            }
        };
        const cancel = () => {
            const replacement = document.createElement('span');
            replacement.className = 'top-title';
            replacement.id = 'conv-title';
            replacement.textContent = currentTitle;
            input.replaceWith(replacement);
            initConvTitleEdit();
        };
        input.addEventListener('keydown', e => {
            if (e.key === 'Enter') { e.preventDefault(); commit(); }
            if (e.key === 'Escape') { e.preventDefault(); cancel(); }
        });
        input.addEventListener('blur', commit);
    });
}
initConvTitleEdit();

/* ----- Session History ----- */
function addSessionToHistory(sessionId, title) {
    const exists = state.sessions.find(s => s.session_id === sessionId);
    if (exists) return;
    state.sessions.unshift({ session_id: sessionId, title });
    renderHistoryList();
}

function renderHistoryList() {
    const list = $('#history-list');
    list.innerHTML = '';
    state.sessions.forEach(s => {
        const div = document.createElement('div');
        div.className = 'history-item' + (s.session_id === state.currentSessionId ? ' active' : '');
        div.dataset.session = s.session_id;
        div.innerHTML = `<span class="history-title">${escapeHtml(s.title)}</span><button class="history-delete-btn" title="删除会话">&times;</button>`;
        div.querySelector('.history-title').addEventListener('click', () => loadSession(s.session_id, s.title));
        div.querySelector('.history-delete-btn').addEventListener('click', (e) => {
            e.stopPropagation();
            deleteSession(s.session_id);
        });
        list.appendChild(div);
    });
}

async function deleteSession(sessionId) {
    showConfirmModal('永久删除对话', '删除后，该对话将不可恢复。确认删除吗？', '删除', async () => {
        await _doDeleteSession(sessionId);
    });
}
async function _doDeleteSession(sessionId) {
    try {
        const uid = state.currentUser?.id;
        const url = uid ? `/sessions/${sessionId}?user_id=${uid}` : `/sessions/${sessionId}`;
        await apiFetch(url, { method: 'DELETE' });
        state.sessions = state.sessions.filter(s => s.session_id !== sessionId);
        if (state.currentSessionId === sessionId) {
            resetToWelcome();
        }
        renderHistoryList();
        showToast('会话已删除');
    } catch (err) {
        showToast('删除失败：' + err.message);
    }
}

async function loadSession(sessionId, title) {
    state.currentSessionId = sessionId;
    enterConversation(sessionId, title);
    $('#messages-area').innerHTML = '';
    highlightActiveSession(sessionId);

    try {
        const data = await apiFetch(`/history/${sessionId}`);
        data.items.forEach(item => {
            appendUserMessage(item.query_content);
            appendAIMessage(item.query_content, item.answer_content);
        });
    } catch (err) {
        showToast('加载历史失败：' + err.message);
    }
}

/* ----- Sidebar Collapse/Expand ----- */
$('#btn-collapse-sidebar').addEventListener('click', () => {
    $('#chat-sidebar').classList.add('collapsed');
    $('#floating-toolbar').style.display = '';
});
$('#btn-expand-sidebar').addEventListener('click', () => {
    $('#chat-sidebar').classList.remove('collapsed');
    $('#floating-toolbar').style.display = 'none';
});
$('#btn-floating-new-chat').addEventListener('click', () => {
    resetToWelcome();
});

/* ----- Chat Event Bindings ----- */
$('#btn-new-chat').addEventListener('click', resetToWelcome);

$('#welcome-send').addEventListener('click', () => {
    const input = $('#welcome-input');
    sendMessage(input.value);
    input.value = '';
});
$('#welcome-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        $('#welcome-send').click();
    }
});

$('#conv-send').addEventListener('click', () => {
    const btn = document.getElementById('conv-send');
    if (btn && btn._isStop) { stopGeneration(); return; }
    const input = $('#conv-input');
    sendMessage(input.value);
    input.value = '';
});
$('#conv-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        $('#conv-send').click();
    }
});

/* ========== Admin Panel ========== */
function enterAdmin() {
    showPage('page-admin');
    loadDocList();
}

/* ----- Admin Nav ----- */
$('#admin-nav').addEventListener('click', e => {
    const item = e.target.closest('.nav-item');
    if (!item) return;
    $$('#admin-nav .nav-item').forEach(n => n.classList.remove('active'));
    item.classList.add('active');
    const tabId = item.dataset.tab;
    $$('.admin-tab').forEach(t => t.classList.remove('active'));
    const tab = document.getElementById(tabId);
    if (tab) tab.classList.add('active');

    const titles = {
        'tab-docs': ['文档管理', '上传 .pdf .docx .doc .xlsx 文件至知识库，系统将自动解析并分类存储'],
        'tab-users': ['用户管理', '查看注册用户、对话历史，注销用户账号'],
        'tab-feedback': ['用户反馈', '查看用户对AI回答的反馈信息'],
        'tab-models': ['模型管理', '查看和配置可用的LLM模型'],
        'tab-crawler': ['爬虫管理', '启动或停止校园新闻爬虫，查看爬取状态和日志'],
        'tab-settings': ['系统设置', '系统基本信息和配置'],
    };
    const t = titles[tabId] || ['', ''];
    $('#admin-page-title').textContent = t[0];
    $('#admin-page-subtitle').textContent = t[1];

    if (tabId === 'tab-docs') loadDocList();
    if (tabId === 'tab-users') loadUserList();
    if (tabId === 'tab-feedback') loadFeedbackList();
    if (tabId === 'tab-models') loadModelList();
    if (tabId === 'tab-crawler') { loadCrawlerStatus(); startCrawlerPolling(); } else { stopCrawlerPolling(); }
});

$('#btn-admin-logout').addEventListener('click', () => {
    state.isAdmin = false;
    showPage('page-login');
});

/* ----- Document Upload ----- */
const uploadArea = $('#upload-area');
const fileInput = $('#file-input');

uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) uploadFile(fileInput.files[0]);
    fileInput.value = '';
});

async function uploadFile(file) {
    showToast('上传中...');
    const formData = new FormData();
    formData.append('file', file);
    try {
        const res = await fetch(API_BASE + '/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'upload failed');
        showToast(`上传成功：${data.file_name}，处理了 ${data.chunks} 个片段`);
        loadDocList();
    } catch (err) {
        showToast('上传失败：' + err.message);
    }
}

/* ----- Document List ----- */
$('#btn-refresh-docs').addEventListener('click', loadDocList);
$('#btn-download-template').addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = API_BASE + '/upload/template';
    a.download = '模板.xlsx';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
});

async function loadDocList() {
    try {
        const data = await apiFetch('/upload/list');
        const body = $('#doc-table-body');
        body.innerHTML = '';
        if (!data.items || data.items.length === 0) {
            body.innerHTML = '<div class="table-row"><div class="td" style="flex:1;justify-content:center;color:var(--text-tertiary)">暂无上传文档</div></div>';
            return;
        }
        data.items.forEach(doc => {
            const row = document.createElement('div');
            row.className = 'table-row';
            row.innerHTML = `
                <div class="td name" style="width:280px">${escapeHtml(doc.file_name)}</div>
                <div class="td" style="width:100px">${escapeHtml(doc.file_type)}</div>
                <div class="td" style="width:120px" title="${escapeHtml(doc.save_path)}">本地存储</div>
                <div class="td" style="width:160px">${formatTime(doc.uploaded_at)}</div>
                <div class="td" style="flex:1;gap:8px"><span class="badge badge--success">已完成</span><button class="admin-action-btn" style="color:var(--danger)" onclick="deleteDoc(${doc.id})">删除</button></div>
            `;
            body.appendChild(row);
        });
    } catch (err) {
        showToast('加载文档列表失败');
    }
}

async function deleteDoc(docId) {
    showConfirmModal('删除文档', '确认删除该文档记录？此操作不可恢复。', '删除', async () => {
        await _doDeleteDoc(docId);
    });
}
async function _doDeleteDoc(docId) {
    try {
        await apiFetch(`/upload/${docId}`, { method: 'DELETE' });
        showToast('已删除');
        loadDocList();
    } catch (err) {
        showToast('删除失败：' + err.message);
    }
}

/* ----- Feedback List ----- */
$('#btn-refresh-feedback').addEventListener('click', loadFeedbackList);

async function loadFeedbackList() {
    try {
        const data = await apiFetch('/admin/feedback');
        const body = $('#feedback-table-body');
        body.innerHTML = '';
        if (!data.items || data.items.length === 0) {
            body.innerHTML = '<div class="table-row"><div class="td" style="flex:1;justify-content:center;color:var(--text-tertiary)">暂无反馈</div></div>';
            return;
        }
        data.items.forEach(fb => {
            const row = document.createElement('div');
            row.className = 'table-row';
            row.dataset.feedbackId = fb.id;
            row.innerHTML = `
                <div class="td" style="width:100px;padding-right:16px">${escapeHtml(fb.user_name || '匿名')}</div>
                <div class="td" style="width:200px;padding-right:24px" title="${escapeHtml(fb.query_content)}">${escapeHtml(truncate(fb.query_content, 30))}</div>
                <div class="td fb-answer-cell" style="width:200px;cursor:pointer;color:var(--primary);text-decoration:underline;text-underline-offset:2px" title="点击查看完整AI回答">${escapeHtml(truncate(fb.answer_content, 28))}</div>
                <div class="td" style="flex:1;padding-left:36px" title="${escapeHtml(fb.feedback_text)}">${escapeHtml(truncate(fb.feedback_text, 40))}</div>
                <div class="td" style="width:160px">${formatTime(fb.created_at)}</div>
                <div class="td" style="width:80px">
                    <button class="admin-action-btn" style="color:var(--danger)" onclick="deleteFeedback(${fb.id}, this)">删除</button>
                </div>
            `;
            row.querySelector('.fb-answer-cell').addEventListener('click', () => {
                showFeedbackDetail(fb.query_content, fb.answer_content, fb.feedback_text);
            });
            body.appendChild(row);
        });
    } catch (err) {
        showToast('加载反馈列表失败');
    }
}

function showFeedbackDetail(query, answer, feedback) {
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:1000;display:flex;align-items:center;justify-content:center;padding:24px';
    const card = document.createElement('div');
    card.style.cssText = 'background:#fff;border-radius:12px;width:100%;max-width:680px;max-height:80vh;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 8px 40px rgba(0,0,0,.18)';
    card.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:space-between;padding:18px 24px;border-bottom:1px solid #eee">
            <span style="font-size:15px;font-weight:600;color:#1a1a1a">反馈详情</span>
            <button id="fb-detail-close" style="background:none;border:none;font-size:20px;cursor:pointer;color:#999;line-height:1">&times;</button>
        </div>
        <div style="flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:16px">
            <div>
                <div style="font-size:11px;font-weight:600;color:#888;text-transform:uppercase;margin-bottom:6px">用户提问</div>
                <div style="font-size:13px;color:#333;line-height:1.7;white-space:pre-wrap;word-break:break-word">${escapeHtml(query)}</div>
            </div>
            <div style="border-top:1px solid #f0f0f0;padding-top:16px">
                <div style="font-size:11px;font-weight:600;color:#888;text-transform:uppercase;margin-bottom:6px">AI 回答</div>
                <div style="font-size:13px;color:#333;line-height:1.7;white-space:pre-wrap;word-break:break-word">${escapeHtml(answer)}</div>
            </div>
            <div style="border-top:1px solid #f0f0f0;padding-top:16px">
                <div style="font-size:11px;font-weight:600;color:#888;text-transform:uppercase;margin-bottom:6px">用户反馈</div>
                <div style="font-size:13px;color:#333;line-height:1.7;white-space:pre-wrap;word-break:break-word">${escapeHtml(feedback)}</div>
            </div>
        </div>
    `;
    overlay.appendChild(card);
    document.body.appendChild(overlay);
    const close = () => overlay.remove();
    card.querySelector('#fb-detail-close').addEventListener('click', close);
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
}

async function deleteFeedback(id, btn) {
    showConfirmModal('删除反馈', '确认删除该条反馈记录？此操作不可撤销。', '删除', async () => {
        btn.disabled = true;
        try {
            await apiFetch(`/admin/feedback/${id}`, { method: 'DELETE' });
            const row = btn.closest('.table-row');
            if (row) row.remove();
            const body = $('#feedback-table-body');
            if (body && body.children.length === 0) {
                body.innerHTML = '<div class="table-row"><div class="td" style="flex:1;justify-content:center;color:var(--text-tertiary)">暂无反馈</div></div>';
            }
            showToast('反馈已删除');
        } catch (err) {
            showToast('删除失败：' + err.message);
            btn.disabled = false;
        }
    });
}

/* ----- Model Management ----- */
$('#btn-add-model').addEventListener('click', () => {
    $('#model-form-card').style.display = '';
    $('#model-form-title').textContent = '添加模型';
    $('#model-form').reset();
    $('#mf-api-base').value = 'https://openrouter.ai/api/v1';
    $('#model-form').removeAttribute('data-edit-id');
});
$('#btn-cancel-model').addEventListener('click', () => {
    $('#model-form-card').style.display = 'none';
});

$('#model-form').addEventListener('submit', async e => {
    e.preventDefault();
    const editId = e.target.getAttribute('data-edit-id');
    const payload = {
        model_name: $('#mf-name').value.trim(),
        model_id: $('#mf-model-id').value.trim(),
        api_base: $('#mf-api-base').value.trim() || 'https://openrouter.ai/api/v1',
        api_key: $('#mf-api-key').value.trim(),
        is_default: 0,
    };
    if (!payload.model_name || !payload.model_id) { showToast('请填写必填字段'); return; }
    try {
        if (editId) {
            await apiFetch(`/models/${editId}`, { method: 'PUT', body: JSON.stringify({ ...payload, enabled: 1 }) });
            showToast('模型已更新');
        } else {
            await apiFetch('/models', { method: 'POST', body: JSON.stringify(payload) });
            showToast('模型添加成功');
        }
        $('#model-form-card').style.display = 'none';
        loadModelList();
    } catch (err) {
        showToast('操作失败：' + err.message);
    }
});

async function loadModelList() {
    try {
        const data = await apiFetch('/models');
        const body = $('#models-table-body');
        body.innerHTML = '';
        if (!data.models || data.models.length === 0) {
            body.innerHTML = '<div class="table-row"><div class="td" style="flex:1;justify-content:center;color:var(--text-tertiary)">暂无模型配置，请添加</div></div>';
            return;
        }
        data.models.forEach(m => {
            const row = document.createElement('div');
            row.className = 'table-row';
            row.innerHTML = `
                <div class="td" style="width:180px">${escapeHtml(m.name)}</div>
                <div class="td" style="width:240px;font-size:12px;color:var(--text-secondary)">${escapeHtml(m.model_id || '')}</div>
                <div class="td" style="flex:1;font-size:12px;color:var(--text-secondary)" title="${escapeHtml(m.api_base || '')}">${escapeHtml(truncate(m.api_base || '', 30))}</div>
                <div class="td" style="width:80px">${m.is_default ? '<span class="badge badge--success">默认</span>' : '<span class="badge badge--warning">备用</span>'}</div>
                <div class="td" style="width:200px;gap:8px">
                    <button class="admin-action-btn" style="color:var(--warning)" onclick="editModel(${m.id})">编辑</button>
                    ${!m.is_default ? `<button class="admin-action-btn" style="color:var(--primary)" onclick="setDefaultModel(${m.id})">设为默认</button>` : ''}
                    <button class="admin-action-btn" style="color:var(--danger)" onclick="deleteModel(${m.id})">删除</button>
                </div>
            `;
            body.appendChild(row);
        });
    } catch (err) {
        showToast('加载模型列表失败');
    }
}

async function editModel(recordId) {
    try {
        const m = await apiFetch(`/models/${recordId}`);
        $('#model-form-card').style.display = '';
        $('#model-form-title').textContent = '编辑模型';
        $('#model-form').setAttribute('data-edit-id', recordId);
        $('#mf-name').value = m.model_name || '';
        $('#mf-model-id').value = m.model_id || '';
        $('#mf-api-base').value = m.api_base || 'https://openrouter.ai/api/v1';
        $('#mf-api-key').value = m.api_key || '';
        $('#model-form-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
        showToast('加载模型信息失败：' + err.message);
    }
}

async function setDefaultModel(recordId) {
    try {
        await apiFetch(`/models/${recordId}/default`, { method: 'POST' });
        showToast('已设为默认模型');
        loadModelList();
    } catch (err) {
        showToast('设置失败：' + err.message);
    }
}

async function deleteModel(recordId) {
    showConfirmModal('删除模型', '确认删除该模型配置？', '删除', async () => {
        await _doDeleteModel(recordId);
    });
}
async function _doDeleteModel(recordId) {
    try {
        await apiFetch(`/models/${recordId}`, { method: 'DELETE' });
        showToast('模型已删除');
        loadModelList();
    } catch (err) {
        showToast('删除失败：' + err.message);
    }
}

/* ========== User Management ========== */
$('#btn-refresh-users').addEventListener('click', loadUserList);
$('#btn-close-user-history').addEventListener('click', () => {
    $('#user-history-section').style.display = 'none';
});
$('#btn-close-session-messages').addEventListener('click', () => {
    $('#session-messages-section').style.display = 'none';
});

async function loadUserList() {
    try {
        const data = await apiFetch('/admin/users');
        const body = $('#users-table-body');
        body.innerHTML = '';
        $('#user-history-section').style.display = 'none';
        if (!data.items || data.items.length === 0) {
            body.innerHTML = '<div class="table-row"><div class="td" style="flex:1;justify-content:center;color:var(--text-tertiary)">暂无注册用户</div></div>';
            return;
        }
        data.items.forEach(u => {
            const row = document.createElement('div');
            row.className = 'table-row';
            row.innerHTML = `
                <div class="td" style="width:60px">${u.id}</div>
                <div class="td" style="width:120px;font-weight:500;color:var(--text-primary)">${escapeHtml(u.name)}</div>
                <div class="td" style="width:150px">${escapeHtml(u.student_id)}</div>
                <div class="td" style="width:140px">${escapeHtml(u.phone)}</div>
                <div class="td" style="width:160px">${formatTime(u.created_at)}</div>
                <div class="td" style="flex:1;gap:8px">
                    <button class="admin-action-btn" style="color:var(--primary)" onclick="viewUserHistory(${u.id}, '${escapeAttr(u.name)}')">对话历史</button>
                    <button class="admin-action-btn" style="color:var(--warning)" onclick="deleteAllUserHistory(${u.id}, '${escapeAttr(u.name)}')">清除历史</button>
                    <button class="admin-action-btn" style="color:var(--danger)" onclick="deleteUser(${u.id}, '${escapeAttr(u.name)}')">注销</button>
                </div>
            `;
            body.appendChild(row);
        });
    } catch (err) {
        showToast('加载用户列表失败：' + err.message);
    }
}

async function viewUserHistory(userId, userName) {
    const section = $('#user-history-section');
    section.style.display = '';
    $('#user-history-title').textContent = `${userName} 的对话历史`;
    $('#session-messages-section').style.display = 'none';
    const body = $('#user-sessions-body');
    body.innerHTML = '<div class="table-row"><div class="td" style="flex:1;justify-content:center;color:var(--text-tertiary)">加载中...</div></div>';
    try {
        const data = await apiFetch(`/admin/users/${userId}/sessions`);
        body.innerHTML = '';
        if (!data.items || data.items.length === 0) {
            body.innerHTML = '<div class="table-row"><div class="td" style="flex:1;justify-content:center;color:var(--text-tertiary)">该用户暂无对话历史</div></div>';
            return;
        }
        data.items.forEach(s => {
            const row = document.createElement('div');
            row.className = 'table-row';
            row.innerHTML = `
                <div class="td" style="width:200px;font-weight:500;color:var(--text-primary)" title="${escapeHtml(s.title)}">${escapeHtml(truncate(s.title, 25))}</div>
                <div class="td" style="width:100px">${s.message_count} 条</div>
                <div class="td" style="width:160px">${formatTime(s.last_active)}</div>
                <div class="td" style="flex:1;gap:8px">
                    <button class="admin-action-btn" style="color:var(--primary)" onclick="viewSessionMessages(${userId}, '${escapeAttr(s.session_id)}', '${escapeAttr(s.title)}')">查看</button>
                    <button class="admin-action-btn" style="color:var(--danger)" onclick="deleteUserSession(${userId}, '${escapeAttr(s.session_id)}')">删除</button>
                </div>
            `;
            body.appendChild(row);
        });
        section.scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
        body.innerHTML = '';
        showToast('加载对话历史失败：' + err.message);
    }
}

async function viewSessionMessages(userId, sessionId, title) {
    const section = $('#session-messages-section');
    section.style.display = '';
    $('#session-messages-title').textContent = `对话详情：${title}`;
    const body = $('#session-messages-body');
    body.innerHTML = '<p style="color:var(--text-tertiary);text-align:center">加载中...</p>';
    try {
        const data = await apiFetch(`/admin/users/${userId}/sessions/${sessionId}/messages`);
        body.innerHTML = '';
        if (!data.items || data.items.length === 0) {
            body.innerHTML = '<p style="color:var(--text-tertiary);text-align:center">无消息记录</p>';
            return;
        }
        data.items.forEach(msg => {
            const qDiv = document.createElement('div');
            qDiv.style.cssText = 'padding:8px 12px;border-radius:8px;background:var(--primary-light);color:var(--text-primary);font-size:13px;align-self:flex-end;max-width:80%';
            qDiv.textContent = msg.query_content;
            body.appendChild(qDiv);

            const aDiv = document.createElement('div');
            aDiv.style.cssText = 'padding:8px 12px;border-radius:8px;background:var(--bg-page);color:var(--text-secondary);font-size:13px;align-self:flex-start;max-width:80%';
            aDiv.textContent = truncate(msg.answer_content, 200);
            body.appendChild(aDiv);
        });
        section.scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
        body.innerHTML = '';
        showToast('加载消息失败：' + err.message);
    }
}

async function deleteUserSession(userId, sessionId) {
    showConfirmModal('永久删除对话', '删除后，该对话将不可恢复。确认删除吗？', '删除', async () => {
        await _doDeleteUserSession(userId, sessionId);
    });
}
async function _doDeleteUserSession(userId, sessionId) {
    try {
        await apiFetch(`/admin/users/${userId}/sessions/${sessionId}`, { method: 'DELETE' });
        showToast('会话已删除');
        viewUserHistory(userId, $('#user-history-title').textContent.replace(' 的对话历史', ''));
    } catch (err) {
        showToast('删除失败：' + err.message);
    }
}

async function deleteAllUserHistory(userId, userName) {
    showConfirmModal('清除对话历史', `确认清除 ${userName} 的全部对话历史？此操作不可恢复。`, '清除', async () => {
        await _doDeleteAllUserHistory(userId, userName);
    });
}
async function _doDeleteAllUserHistory(userId, userName) {
    try {
        await apiFetch(`/admin/users/${userId}/history`, { method: 'DELETE' });
        showToast(`${userName} 的对话历史已清除`);
        $('#user-history-section').style.display = 'none';
    } catch (err) {
        showToast('清除失败：' + err.message);
    }
}

async function deleteUser(userId, userName) {
    showConfirmModal('注销用户', `确认注销用户 ${userName}？将同时删除该用户的全部数据，此操作不可恢复！`, '注销', async () => {
        await _doDeleteUser(userId, userName);
    });
}
async function _doDeleteUser(userId, userName) {
    try {
        await apiFetch(`/admin/users/${userId}`, { method: 'DELETE' });
        showToast(`用户 ${userName} 已注销`);
        loadUserList();
    } catch (err) {
        showToast('注销失败：' + err.message);
    }
}

/* ========== Crawler Management ========== */
let _crawlerPollTimer = null;

function startCrawlerPolling() {
    stopCrawlerPolling();
    _crawlerPollTimer = setInterval(loadCrawlerStatus, 3000);
}

function stopCrawlerPolling() {
    if (_crawlerPollTimer) {
        clearInterval(_crawlerPollTimer);
        _crawlerPollTimer = null;
    }
}

function formatUptime(seconds) {
    if (seconds == null) return '-';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    const parts = [];
    if (h > 0) parts.push(h + '时');
    if (m > 0) parts.push(m + '分');
    parts.push(s + '秒');
    return parts.join('');
}

async function loadCrawlerStatus() {
    try {
        const data = await apiFetch('/crawler/status');
        const running = data.running;
        const dot = $('#crawler-status-dot');
        const text = $('#crawler-status-text');
        const btnStart = $('#btn-crawler-start');
        const btnStop = $('#btn-crawler-stop');

        dot.className = 'crawler-status-dot ' + (running ? 'dot-running' : 'dot-stopped');
        text.textContent = running ? '运行中' : '已停止';
        text.className = 'crawler-status-text ' + (running ? 'text-running' : 'text-stopped');

        btnStart.disabled = running;
        btnStop.disabled = !running;

        $('#crawler-pid').textContent = data.pid || '-';
        $('#crawler-uptime').textContent = formatUptime(data.uptime_seconds);

        const logArea = $('#crawler-log-area');
        if (data.recent_logs && data.recent_logs.length > 0) {
            logArea.innerHTML = data.recent_logs.map(l => '<div class="crawler-log-line">' + escapeHtml(l) + '</div>').join('');
            logArea.scrollTop = logArea.scrollHeight;
        } else if (!running) {
            logArea.innerHTML = '<p class="crawler-log-placeholder">暂无日志输出</p>';
        }
    } catch (err) {
        $('#crawler-status-text').textContent = '状态获取失败';
    }
}

$('#btn-crawler-start').addEventListener('click', async () => {
    const btn = $('#btn-crawler-start');
    btn.disabled = true;
    btn.textContent = '启动中...';
    try {
        const data = await apiFetch('/crawler/start', { method: 'POST' });
        showToast(data.message);
        await loadCrawlerStatus();
    } catch (err) {
        showToast('启动失败：' + err.message);
    } finally {
        btn.textContent = '启动爬虫';
    }
});

$('#btn-crawler-stop').addEventListener('click', async () => {
    if (!confirm('确认停止爬虫？')) return;
    const btn = $('#btn-crawler-stop');
    btn.disabled = true;
    btn.textContent = '停止中...';
    try {
        const data = await apiFetch('/crawler/stop', { method: 'POST' });
        showToast(data.message);
        await loadCrawlerStatus();
    } catch (err) {
        showToast('停止失败：' + err.message);
    } finally {
        btn.textContent = '停止爬虫';
    }
});

$('#btn-refresh-crawler').addEventListener('click', loadCrawlerStatus);

/* ---------- Utilities ---------- */
function formatTime(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    const pad = n => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.slice(0, len) + '...' : str;
}

/* ---------- Init ---------- */
(function initFromSession() {
    const savedUser = sessionStorage.getItem('campusqa_user');
    const savedPage = sessionStorage.getItem('campusqa_page');
    if (savedUser && savedPage === 'page-chat') {
        try {
            state.currentUser = JSON.parse(savedUser);
            enterChat();
            return;
        } catch (e) { /* fall through to login */ }
    }
    showPage('page-login');
})();
