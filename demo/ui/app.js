/* global window, document */
const $ = (id) => document.getElementById(id);

const state = {
  profile: null,
  language: "English",
  lastItem: null,
  ttsAudio: null,
  typeaheadOpen: false,
};

function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    if (t) clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", "\"":"&quot;", "'":"&#39;"
  }[c]));
}

function gradientForSeed(seed) {
  // deterministic-ish gradient
  const n = Number(seed) || 0;
  const a = (n * 37) % 360;
  const b = (a + 50) % 360;
  return `linear-gradient(135deg, hsl(${a} 70% 45%) 0%, hsl(${b} 70% 35%) 100%)`;
}

async function apiGet(path) {
  const r = await fetch(path, { headers: { "accept":"application/json" } });
  if (!r.ok) throw new Error(`${path} failed: ${r.status}`);
  return await r.json();
}

async function apiPost(path, body, extraHeaders = {}) {
  const r = await fetch(path, {
    method:"POST",
    headers: { "content-type":"application/json", ...extraHeaders },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`${path} failed: ${r.status} ${txt}`);
  }
  return await r.json();
}

function setNavbarProfile(profile) {
  $("profileIcon").textContent = profile.displayName.charAt(0);
  $("profileIcon").className = `profile-icon ${profile.name}`;
  $("userName").textContent = profile.displayName;
}

function show(el) { el.style.display = "block"; }
function hide(el) { el.style.display = "none"; }

function openHow() { $("howModal").classList.add("active"); }
function closeHow() { $("howModal").classList.remove("active"); }

function openModal() { $("videoModal").classList.add("active"); }
function closeModal() {
  $("videoModal").classList.remove("active");
  const v = $("videoPlayer");
  v.pause();
  v.currentTime = 0;
  stopTTS();
}

function setStatusPills(ok, liftText) {
  $("statusPill").textContent = ok ? "API: ready" : "API: down";
  $("liftPill").textContent = liftText || "Lift: —";
  $("howApi").textContent = ok ? "API: ready" : "API: down";
  $("howLift").textContent = liftText || "Lift: —";
  $("howLang").textContent = `Language: ${state.language}`;
}

async function refreshMeta() {
  let ok = false;
  try {
    const h = await apiGet("/health");
    ok = !!h.ready;
  } catch (_) { ok = false; }

  let liftText = "Lift: —";
  try {
    const l = await apiGet("/metrics/lift");
    if (l.ok) liftText = `Lift nDCG@10: +${Number(l.lift).toFixed(4)} (LTR vs hybrid)`;
  } catch (_) { /* ignore */ }

  setStatusPills(ok, liftText);
}

function renderRows(rows) {
  const root = $("contentSection");
  root.innerHTML = "";

  rows.forEach((row) => {
    const block = document.createElement("div");
    block.className = "row-block";

    const title = document.createElement("h2");
    title.className = "row-title";
    title.textContent = row.title;

    const items = document.createElement("div");
    items.className = "row-items";

    (row.items || []).forEach((it) => {
      const card = document.createElement("div");
      card.className = "item";
      card.style.background = it.poster || gradientForSeed(it.doc_id);

      const center = document.createElement("div");
      center.className = "item-center";
      center.textContent = it.title;

      const overlay = document.createElement("div");
      overlay.className = "item-overlay";
      overlay.innerHTML = `
        <div class="item-title">${esc(it.title)}</div>
        <div class="item-rating">${esc(it.match || "")}</div>
      `;

      card.appendChild(center);
      card.appendChild(overlay);

      card.onclick = () => openItem(it);
      items.appendChild(card);
    });

    block.appendChild(title);
    block.appendChild(items);
    root.appendChild(block);
  });
}

async function loadLanguages() {
  const sel = $("languageSelect");
  try {
    const j = await apiGet("/languages");
    const langs = j.languages || [];
    sel.innerHTML = "";
    langs.forEach((l) => {
      const o = document.createElement("option");
      o.value = l;
      o.textContent = l;
      sel.appendChild(o);
    });
    sel.value = state.language;
  } catch (_) {
    // keep default
  }

  sel.onchange = async () => {
    state.language = sel.value;
    await refreshMeta();
    await loadFeed();
  };
}

async function loadFeed() {
  if (!state.profile) return;
  try {
    const j = await apiGet(`/feed?profile=${encodeURIComponent(state.profile.name)}&language=${encodeURIComponent(state.language)}&rows=6&k=12`);
    renderRows(j.rows || []);
  } catch (e) {
    $("contentSection").innerHTML = `<p style="padding:40px;color:#bbb">Feed error: ${esc(e.message)}</p>`;
  }
}

async function doSearch(query) {
  if (!query) return;

  const body = {
    query,
    method: "hybrid_ltr",
    k: 20,
    candidate_k: 200,
    rerank_k: 50,
    alpha: 0.5,
    debug: false,
  };

  const j = await apiPost("/search", body, { "X-Language": state.language });

  const items = (j.hits || []).map((h) => ({
    doc_id: h.doc_id,
    title: h.title || "(untitled)",
    match: `score ${Number(h.score).toFixed(3)}`,
    poster: gradientForSeed(h.doc_id),
  }));

  renderRows([{ title: `Search results for "${query}"`, items }]);
}

function setTypeahead(items) {
  const box = $("typeahead");
  if (!items || items.length === 0) {
    box.classList.remove("active");
    box.innerHTML = "";
    state.typeaheadOpen = false;
    return;
  }

  box.innerHTML = "";
  items.forEach((t) => {
    const div = document.createElement("div");
    div.className = "row";
    div.innerHTML = `<div>${esc(t)}</div><div class="small">title match</div>`;
    div.onclick = async () => {
      $("searchInput").value = t;
      setTypeahead([]);
      await doSearch(t);
    };
    box.appendChild(div);
  });

  box.classList.add("active");
  state.typeaheadOpen = true;
}

const typeaheadFetch = debounce(async () => {
  const q = $("searchInput").value.trim();
  if (q.length < 2) {
    setTypeahead([]);
    return;
  }
  try {
    const j = await apiGet(`/suggest?q=${encodeURIComponent(q)}&n=8&profile=${encodeURIComponent(state.profile?.name || "chrisen")}`);
    setTypeahead(j.suggestions || []);
  } catch (_) {
    setTypeahead([]);
  }
}, 150);

async function openItem(it) {
  state.lastItem = it;
  openModal();

  $("modalTitle").textContent = it.title;
  $("modalDescription").textContent = "Loading details…";
  $("modalGenres").textContent = "";
  $("modalMatch").textContent = it.match || "—";
  $("modalYear").textContent = "—";
  $("modalRating").textContent = "—";
  $("modalDuration").textContent = "—";

  $("explainBody").textContent = "Click “Why recommended?”";
  $("debugPre").textContent = "—";
  $("explainMeta").textContent = "—";

  try {
    const meta = await apiGet(`/item?doc_id=${encodeURIComponent(it.doc_id)}&language=${encodeURIComponent(state.language)}`);
    $("modalDescription").textContent = meta.synopsis || meta.text || "—";
    $("modalGenres").innerHTML = `<strong>Genres:</strong> ${esc(meta.genres || "unknown")} &nbsp; <strong>Tags:</strong> ${esc(meta.tags || "none")} &nbsp; <strong>Language:</strong> ${esc(meta.language || state.language)}`;
    $("modalYear").textContent = meta.year || "—";
    $("modalRating").textContent = meta.rating || "—";
    $("modalDuration").textContent = meta.duration || "—";
    $("modalMatch").textContent = meta.match || it.match || "—";
  } catch (e) {
    $("modalDescription").textContent = `Failed to load item metadata: ${e.message}`;
  }
}

async function explainCurrent(agentic) {
  if (!state.lastItem || !state.profile) return;
  $("explainBody").textContent = agentic ? "Agentic explanation loading…" : "Explanation loading…";
  $("explainMeta").textContent = "—";
  $("debugPre").textContent = "—";

  try {
    const j = await apiGet(
      `/explain?doc_id=${encodeURIComponent(state.lastItem.doc_id)}&profile=${encodeURIComponent(state.profile.name)}&language=${encodeURIComponent(state.language)}&agentic=${agentic ? "true" : "false"}`
    );
    $("explainBody").textContent = j.answer || "—";
    $("explainMeta").textContent = j.warning ? `warning: ${j.warning}` : `citations: ${(j.citations || []).join(", ") || "—"}`;

    // keep debug in a collapsible block only
    $("debugPre").textContent = JSON.stringify(j, null, 2);
  } catch (e) {
    $("explainBody").textContent = `Explain failed: ${e.message}`;
  }
}

/* Server-side TTS */
async function speakText(text) {
  const audio = $("ttsAudio");
  state.ttsAudio = audio;

  const q = encodeURIComponent(text.slice(0, 400));
  const lang = encodeURIComponent(state.language);

  try {
    const url = `/tts?lang=${lang}&text=${q}`;
    audio.src = url;
    await audio.play();
  } catch (e) {
    // if browser blocks autoplay, user must click again
    console.warn("TTS play failed:", e);
  }
}

function stopTTS() {
  const audio = $("ttsAudio");
  audio.pause();
  audio.currentTime = 0;
}

/* Events */
document.addEventListener("DOMContentLoaded", async () => {
  // Login flow
  $("loginForm").addEventListener("submit", (e) => {
    e.preventDefault();
    $("loginScreen").classList.add("hidden");
    $("userSelectionScreen").classList.add("active");
  });

  // Profile selection
  document.querySelectorAll(".user-card").forEach((el) => {
    el.addEventListener("click", async () => {
      const name = el.getAttribute("data-user");
      state.profile = { name, displayName: name.charAt(0).toUpperCase() + name.slice(1) };

      $("userSelectionScreen").classList.remove("active");
      show($("mainApp"));
      setNavbarProfile(state.profile);

      await loadLanguages();
      await refreshMeta();
      await loadFeed();
    });
  });

  // Navbar scroll effect
  window.addEventListener("scroll", () => {
    const navbar = $("navbar");
    if (window.scrollY > 50) navbar.classList.add("scrolled");
    else navbar.classList.remove("scrolled");
  });

  // Search
  $("searchInput").addEventListener("input", typeaheadFetch);
  $("searchInput").addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
      const q = $("searchInput").value.trim();
      setTypeahead([]);
      await doSearch(q);
    }
    if (e.key === "Escape") setTypeahead([]);
  });

  document.addEventListener("click", (e) => {
    if (!state.typeaheadOpen) return;
    const box = $("typeahead");
    if (!box.contains(e.target) && e.target !== $("searchInput")) setTypeahead([]);
  });

  // Switch profiles
  $("userProfile").addEventListener("click", async () => {
    const ok = window.confirm("Switch profiles?");
    if (!ok) return;
    hide($("mainApp"));
    $("userSelectionScreen").classList.add("active");
    setTypeahead([]);
    stopTTS();
  });

  // Hero buttons
  $("heroPlayBtn").onclick = async () => {
    $("searchInput").value = "tamil love story";
    setTypeahead([]);
    await doSearch("tamil love story");
    window.scrollTo({ top: $("contentSection").offsetTop - 60, behavior:"smooth" });
  };
  $("heroInfoBtn").onclick = openHow;
  $("logoBtn").onclick = openHow;
  $("howBtn").onclick = openHow;
  $("howCloseBtn").onclick = closeHow;

  // Modal events
  $("closeModalBtn").onclick = closeModal;
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeModal();
      closeHow();
      setTypeahead([]);
    }
  });

  $("whyBtn").onclick = () => explainCurrent(false);
  $("agentWhyBtn").onclick = () => explainCurrent(true);
  $("speakBtn").onclick = async () => {
    const text = $("explainBody").textContent || $("modalTitle").textContent;
    await speakText(text);
  };
  $("stopSpeakBtn").onclick = stopTTS;

  // initial hidden
  hide($("mainApp"));
});
