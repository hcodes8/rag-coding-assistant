const BASE = window.location.protocol === 'file:' ? 'http://127.0.0.1:5000' : '';
let activeLanguage = null;
let isStreaming = false;

const $ = id => document.getElementById(id);

function setStatus(state, label) {
    const dot = $('status-dot');
    dot.className = '';
    if (state) dot.classList.add(state);
    $('status-label').textContent = label;
}

function autoResize(ta) {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 140) + 'px';
}

function scrollBottom() {
    const m = $('messages');
    m.scrollTop = m.scrollHeight;
}

function addMessage(role, text) {
    const empty = $('empty-state');
    if (empty) empty.remove();

    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const div = document.createElement('div');
    div.className = `msg ${role}`;

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = `${role === 'user' ? 'You' : 'Assistant'} · ${now}`;

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    if (role === 'assistant') {
        bubble.innerHTML = renderMarkdown(text);
    } else {
        bubble.textContent = text;
    }

    const copyBtn = document.createElement('button');
    copyBtn.className = 'msg-copy';
    copyBtn.textContent = 'copy';
    copyBtn.addEventListener('click', () => copyToClipboard(text, copyBtn));

    div.appendChild(meta);
    div.appendChild(bubble);
    div.appendChild(copyBtn);
    $('messages').appendChild(div);
    scrollBottom();
    return bubble;
}

function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Markdown Rendering
function renderMarkdown(raw) {
    const codeBlocks = [];
    let s = raw.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
        const idx = codeBlocks.length;
        codeBlocks.push({ lang: lang || '', code });
        return `\x00CODE${idx}\x00`;
    });

    s = s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    const lines = s.split('\n');
    const out = [];
    let listStack = [];

    function closeLists() {
        while (listStack.length) { out.push(`</${listStack.pop()}>`); }
    }

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        if (/^\x00CODE\d+\x00$/.test(line.trim())) {
            closeLists();
            const idx = parseInt(line.trim().match(/\d+/)[0]);
            const { lang, code } = codeBlocks[idx];
            const escaped = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            out.push(
                `<pre><button class="pre-copy" onclick="copyCode(this)">copy</button>` +
                `<code class="language-${escHtml(lang)}">${escaped}</code></pre>`
            );
            continue;
        }

        if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) {
            closeLists(); out.push('<hr>'); continue;
        }

        const h = line.match(/^(#{1,3})\s+(.*)/);
        if (h) {
            closeLists();
            const lvl = h[1].length;
            out.push(`<h${lvl}>${inlineMarkdown(h[2])}</h${lvl}>`);
            continue;
        }

        if (/^>\s?/.test(line)) {
            closeLists();
            out.push(`<blockquote>${inlineMarkdown(line.replace(/^>\s?/, ''))}</blockquote>`);
            continue;
        }

        const ul = line.match(/^(\s*)[*\-+]\s+(.*)/);
        if (ul) {
            if (!listStack.length || listStack[listStack.length - 1] !== 'ul') {
                closeLists(); out.push('<ul>'); listStack.push('ul');
            }
            out.push(`<li>${inlineMarkdown(ul[2])}</li>`);
            continue;
        }

        const ol = line.match(/^(\s*)\d+\.\s+(.*)/);
        if (ol) {
            if (!listStack.length || listStack[listStack.length - 1] !== 'ol') {
                closeLists(); out.push('<ol>'); listStack.push('ol');
            }
            out.push(`<li>${inlineMarkdown(ol[2])}</li>`);
            continue;
        }

        if (line.trim() === '') {
            closeLists(); out.push(''); continue;
        }
        closeLists();
        out.push(inlineMarkdown(line));
    }
    closeLists();

    const html = out.join('\n')
        .replace(/(^|\n)(?!<[hup\x00]|<ol|<bl|<hr|<pre|<blockquote)([^\n]+)/g, (m, pre, content) => {
            if (!content.trim()) return m;
            return `${pre}<p>${content}</p>`;
        });

    return html;
}

function inlineMarkdown(s) {
    s = s.replace(/\*\*(.+?)\*\*|__(.+?)__/g, (_, a, b) => `<strong>${a || b}</strong>`);
    s = s.replace(/\*(.+?)\*/g, (_, a) => `<em>${a}</em>`);
    s = s.replace(/(?<!\w)_(.+?)_(?!\w)/g, (_, a) => `<em>${a}</em>`);
    s = s.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
    return s;
}

function copyCode(btn) {
    const code = btn.nextElementSibling.textContent;
    copyToClipboard(code, btn);
}

function copyToClipboard(text, btn) {
    navigator.clipboard.writeText(text).then(() => {
        const orig = btn.textContent;
        btn.textContent = 'copied';
        btn.classList.add('copied');
        setTimeout(() => { btn.textContent = orig; btn.classList.remove('copied'); }, 1800);
    });
}

async function loadLanguages() {
    try {
        const r = await fetch(`${BASE}/api/languages`);
        const d = await r.json();
        renderLanguages(d.languages || []);
        pollStatus();
    } catch {
        setStatus('error', 'server offline');
        setTimeout(loadLanguages, 3000);
    }
}

function renderLanguages(langs) {
    const list = $('lang-list');
    list.innerHTML = '';
    langs.forEach(lang => {
        const btn = document.createElement('button');
        btn.className = 'lang-btn';
        btn.dataset.lang = lang;
        btn.innerHTML = `<span class="dot"></span>${escHtml(lang)}`;
        btn.addEventListener('click', () => activateLanguage(lang, btn));
        list.appendChild(btn);
    });
}

async function activateLanguage(lang, btn) {
    if (isStreaming) return;
    document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('loading-lang');
    const origHtml = btn.innerHTML;
    btn.innerHTML = `<span class="spinner"></span>${escHtml(lang)}`;
    setStatus('loading', `loading ${lang}…`);

    try {
        await fetch(`${BASE}/api/activate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: lang })
        });
        activeLanguage = lang;
        btn.classList.remove('loading-lang');
        btn.classList.add('active');
        btn.innerHTML = `<span class="dot"></span>${escHtml(lang)}`;
        setStatus('ready', `${lang} · ready`);
        $('send-btn').disabled = false;
        $('no-lang-warning').style.display = 'none';
        $('sidebar-footer').textContent = `Active: ${lang}`;
    } catch (e) {
        btn.classList.remove('loading-lang');
        btn.innerHTML = origHtml;
        setStatus('error', 'activation failed');
    }
}

async function pollStatus() {
    try {
        const r = await fetch(`${BASE}/api/status`);
        const d = await r.json();
        if (d.ready) {
            activeLanguage = d.language;
            setStatus('ready', `${d.language} · ready`);
            $('send-btn').disabled = false;
            document.querySelectorAll('.lang-btn').forEach(b => {
                b.classList.toggle('active', b.dataset.lang === d.language);
            });
            $('sidebar-footer').textContent = `Active: ${d.language}`;
        } else {
            setStatus(null, 'no language loaded');
        }
    } catch {
        setStatus('error', 'server offline');
    }
}

async function sendQuestion() {
    const input = $('question-input');
    const question = input.value.trim();
    if (!question || isStreaming) return;
    if (!activeLanguage) {
        $('no-lang-warning').style.display = 'block';
        return;
    }

    isStreaming = true;
    $('send-btn').disabled = true;
    input.value = '';
    autoResize(input);

    addMessage('user', question);

    const empty = $('empty-state');
    if (empty) empty.remove();
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const aDiv = document.createElement('div');
    aDiv.className = 'msg assistant';
    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = `Assistant · ${now}`;
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = '<span class="cursor"></span>';
    aDiv.appendChild(meta);
    aDiv.appendChild(bubble);
    $('messages').appendChild(aDiv);
    scrollBottom();

    let fullText = '';
    let done = false;

    try {
        const response = await fetch(`${BASE}/api/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (!done) {
            const { done: readerDone, value } = await reader.read();
            if (readerDone) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data:')) continue;
                const raw = line.slice(5).trim();
                if (raw === '[DONE]') { done = true; break; }
                try {
                    const obj = JSON.parse(raw);
                    fullText += obj.token;
                    bubble.innerHTML = renderMarkdown(fullText) + '<span class="cursor"></span>';
                    scrollBottom();
                } catch { }
            }
        }
    } catch (e) {
        fullText = 'Error: could not reach server.';
    }

    bubble.innerHTML = renderMarkdown(fullText);
    const copyBtn = document.createElement('button');
    copyBtn.className = 'msg-copy';
    copyBtn.textContent = 'copy';
    copyBtn.addEventListener('click', () => copyToClipboard(fullText, copyBtn));
    aDiv.appendChild(copyBtn);
    isStreaming = false;
    $('send-btn').disabled = false;
    scrollBottom();
}

const ta = $('question-input');
ta.addEventListener('input', () => autoResize(ta));
ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
});
$('send-btn').addEventListener('click', sendQuestion);

loadLanguages();
