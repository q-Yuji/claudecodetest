// Jarvis AI Chat — loaded by morning_brief.html
// Depends on: D and BT globals defined in the parent HTML script block
// Talks to: http://localhost:8765 (chat_server.py)

const _AI = 'http://localhost:8765';
let _msgs = [];
let _busy  = false;

// ── server health ────────────────────────────────────────────────────────────
async function aiCheck() {
  const warn = document.getElementById('ai-warn');
  try {
    await fetch(_AI + '/health', { signal: AbortSignal.timeout(2000) });
    if (warn) warn.style.display = 'none';
    return true;
  } catch {
    if (warn) warn.style.display = 'block';
    return false;
  }
}

// ── key form ─────────────────────────────────────────────────────────────────
function aiToggleKey() {
  const el = document.getElementById('ai-key-form');
  if (!el) return;
  const show = el.style.display !== 'block';
  el.style.display = show ? 'block' : 'none';
  if (show) setTimeout(() => document.getElementById('ai-key-input')?.focus(), 40);
}

async function aiSaveKey() {
  const key = (document.getElementById('ai-key-input')?.value || '').trim();
  if (!key) return;
  try {
    const r = await fetch(_AI + '/set-key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key }),
    });
    if (r.ok) {
      document.getElementById('ai-key-form').style.display = 'none';
      document.getElementById('ai-key-input').value = '';
      _bubble('system', 'API key saved.');
    }
  } catch {
    _bubble('system', 'Cannot reach chat server — is python chat_server.py running?');
  }
}

// ── message rendering ─────────────────────────────────────────────────────────
function _bubble(role, text) {
  const box   = document.getElementById('ai-msgs');
  if (!box) return null;
  const wrap  = document.createElement('div');
  wrap.className = 'chat-msg ' + (role === 'user' ? 'user' : 'ai');
  const label = role === 'user' ? 'You' : role === 'system' ? 'System' : 'Jarvis';
  const bub   = document.createElement('div');
  bub.className = 'chat-bubble';
  bub.textContent = text;
  wrap.innerHTML  = '<div class="chat-label">' + label + '</div>';
  wrap.appendChild(bub);
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
  return bub;
}

function _typing(show) {
  const existing = document.getElementById('ai-typing');
  if (!show) { existing?.remove(); return; }
  if (existing) return;
  const box  = document.getElementById('ai-msgs');
  if (!box) return;
  const div  = document.createElement('div');
  div.id     = 'ai-typing';
  div.className = 'chat-msg ai';
  div.innerHTML  = '<div class="chat-label">Jarvis</div>'
    + '<div class="chat-bubble">'
    + '<span class="typing-dot"></span>'
    + '<span class="typing-dot"></span>'
    + '<span class="typing-dot"></span>'
    + '</div>';
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

// ── send ──────────────────────────────────────────────────────────────────────
async function aiSend() {
  if (_busy) return;
  const inp = document.getElementById('ai-input');
  if (!inp) return;
  const text = inp.value.trim();
  if (!text) return;

  if (!await aiCheck()) return;

  inp.value = '';
  _msgs.push({ role: 'user', content: text });
  _bubble('user', text);
  _typing(true);

  _busy = true;
  const btn = document.getElementById('ai-send');
  if (btn) btn.disabled = true;

  let bub      = null;
  let fullText = '';

  try {
    const resp = await fetch(_AI + '/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ messages: _msgs }),
    });

    _typing(false);
    bub = _bubble('ai', '');

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let   buf     = '';

    outer: while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const d = line.slice(6);
        if (d === '[DONE]') { reader.cancel(); break outer; }
        try {
          const { text: chunk } = JSON.parse(d);
          if (chunk) {
            fullText += chunk;
            if (bub) bub.textContent = fullText;
            const box = document.getElementById('ai-msgs');
            if (box) box.scrollTop = box.scrollHeight;
          }
        } catch {}
      }
    }

    if (fullText) _msgs.push({ role: 'assistant', content: fullText });
  } catch (e) {
    _typing(false);
    _bubble('system', 'Error: ' + e.message);
  }

  _busy = false;
  if (btn) btn.disabled = false;
  inp.focus();
}

// ── init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Ctrl+Enter / Cmd+Enter to send
  document.getElementById('ai-input')?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      aiSend();
    }
  });

  // Greeting with today's context
  setTimeout(() => {
    const today = (typeof BT !== 'undefined' && BT.length) ? BT[BT.length - 1] : null;
    const nq    = (typeof D !== 'undefined') ? D?.ctx?.NQ?.last : null;

    if (today) {
      const ah = today.asia_high, al = today.asia_low;
      const lh = today.london_high, ll = today.london_low;
      let sweep = 'London inside Asia range — manipulation phase not yet';
      if (lh > ah) sweep = 'London swept Asia High (' + lh + ' > ' + ah + ') → watch for NY bearish distribution';
      else if (ll < al) sweep = 'London swept Asia Low (' + ll + ' < ' + al + ') → watch for NY bullish distribution';

      const lines = [
        'Ready. Today’s context loaded:',
        '  Session : ' + sweep,
        '  Bias    : ' + today.day_bias,
        '  Asia    : ' + ah + '–' + al,
        '  London  : ' + lh + '–' + ll,
        nq ? '  NQ last : ' + nq : null,
        '',
        'Ask me about GEX levels, trade setups, session structure, or anything else.',
      ].filter(Boolean);

      _bubble('ai', lines.join('\n'));
    } else {
      _bubble('ai', 'Ready. Ask me about GEX levels, trade setups, or session structure.');
    }
  }, 250);
});
