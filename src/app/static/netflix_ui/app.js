const $ = (id) => document.getElementById(id);

const qEl = $("q");
const outEl = $("out");
const srcEl = $("sources");
const statusEl = $("status");
const langEl = $("lang");
const goBtn = $("go");
const micBtn = $("mic");
const speakBtn = $("speak");

let lastAnswer = "";
let recognizing = false;
let recog = null;

function setStatus(msg) { statusEl.textContent = msg || ""; }
function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;" }[c]));
}

function renderAnswer(answer, sources) {
  lastAnswer = answer || "";
  outEl.innerHTML = `<h3>Answer</h3><div>${escapeHtml(answer)}</div>`;

  if (!sources || !sources.length) {
    srcEl.innerHTML = "";
    return;
  }
  const cards = sources.map((s) => {
    const title = escapeHtml(s.title || s.source || "Source");
    const snippet = escapeHtml(s.snippet || s.text || "");
    return `<div class="src"><div><b>${title}</b></div><div>${snippet}</div></div>`;
  }).join("");
  srcEl.innerHTML = `<h3>Sources</h3>${cards}`;
}

async function ask() {
  const q = qEl.value.trim();
  if (!q) return;

  goBtn.disabled = true;
  setStatus("Thinking…");
  outEl.innerHTML = "";
  srcEl.innerHTML = "";

  const lang = langEl.value; // "auto" or code
  const body = { question: q };
  if (lang !== "auto") body.lang = lang;

  try {
    const r = await fetch("/answer", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();

    // adapt if your response shape differs
    renderAnswer(j.answer || j.response || "", j.sources || j.hits || []);
    setStatus("");
  } catch (e) {
    setStatus(`Error: ${e.message}`);
  } finally {
    goBtn.disabled = false;
  }
}

goBtn.addEventListener("click", ask);
qEl.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") ask();
});

// Text-to-speech
speakBtn.addEventListener("click", () => {
  if (!lastAnswer) return;
  if (!("speechSynthesis" in window)) {
    alert("Speech synthesis not supported in this browser.");
    return;
  }
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(lastAnswer);

  const lang = langEl.value;
  if (lang !== "auto") u.lang = lang;

  window.speechSynthesis.speak(u);
});

// Speech-to-text (best effort)
function initSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    micBtn.disabled = true;
    micBtn.title = "SpeechRecognition not supported (try Chrome)";
    return;
  }

  recog = new SR();
  recog.continuous = false;
  recog.interimResults = false;

  recog.onstart = () => {
    recognizing = true;
    micBtn.textContent = "🛑";
    setStatus("Listening…");
  };
  recog.onend = () => {
    recognizing = false;
    micBtn.textContent = "🎙️";
    setStatus("");
  };
  recog.onerror = (e) => {
    setStatus(`Mic error: ${e.error || "unknown"}`);
  };
  recog.onresult = (e) => {
    const text = e.results?.[0]?.[0]?.transcript || "";
    if (text) qEl.value = text;
  };

  micBtn.addEventListener("click", () => {
    if (!recog) return;
    const lang = langEl.value;
    if (lang !== "auto") recog.lang = lang;

    try {
      if (recognizing) recog.stop();
      else recog.start();
    } catch (err) {
      setStatus("Mic failed (Safari can be flaky). Try Chrome.");
    }
  });
}

initSpeech();
