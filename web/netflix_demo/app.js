const $ = (id) => document.getElementById(id);

let currentUser = null;
let currentQuery = "";
let currentSelected = null; // {doc_id,title,text,score}
let lastSpoken = "";

const ROWS = [
  { title: "Trending now", query: "action" },
  { title: "Gritty crime & thrillers", query: "gritty crime thriller" },
  { title: "Feel-good comedies", query: "feel good comedy" },
  { title: "Sci-Fi & mind-bending", query: "sci fi mind bending" },
  { title: "Animation & family", query: "animation family" },
  { title: "Romance & drama", query: "romantic drama" },
];

function hashColor(str) {
  let h = 0;
  for (let i=0;i<str.length;i++) h = (h*31 + str.charCodeAt(i)) >>> 0;
  const a = (h % 360);
  const b = ((h >>> 8) % 360);
  return `linear-gradient(135deg, hsl(${a} 70% 45%), hsl(${b} 70% 35%))`;
}

function parseMeta(text) {
  // text: "Title: X | Genres: A,B | Tags: t1,t2"
  const out = { genres:"—", tags:"—" };
  if (!text) return out;
  const parts = text.split("|").map(s => s.trim());
  for (const p of parts) {
    if (p.toLowerCase().startsWith("genres:")) out.genres = p.slice(7).trim() || "—";
    if (p.toLowerCase().startsWith("tags:")) out.tags = p.slice(5).trim() || "—";
  }
  return out;
}

async function apiHealth() {
  try {
    const h = await fetch("/health").then(r=>r.json());
    $("apiPill").innerHTML = `API: ${h.ready ? '<span class="ok">ready</span>' : '<span class="bad">not-ready</span>'}`;
  } catch {
    $("apiPill").innerHTML = `API: <span class="bad">down</span>`;
  }
}

async function apiLift() {
  try {
    const j = await fetch("/metrics/lift").then(r=>r.json());
    if (j.ok) {
      const lift = Number(j.lift).toFixed(4);
      $("liftPill").innerHTML = `Offline lift: <b class="ok">+${lift} nDCG@10</b> (hybrid_ltr − hybrid)`;
    } else {
      $("liftPill").textContent = "Offline lift: —";
    }
  } catch {
    $("liftPill").textContent = "Offline lift: —";
  }
}

async function apiLatency() {
  try {
    const j = await fetch("/metrics/latency").then(r=>r.json());
    const p95 = j?.p95_ms?.["/search"];
    if (p95 !== undefined && p95 !== null) $("latKpi").textContent = `p95 /search: ${Number(p95).toFixed(1)}ms`;
  } catch {
    // ignore
  }
}

async function typeahead(q) {
  if (!q || q.length < 2) {
    $("typeahead").classList.add("hidden");
    $("typeahead").innerHTML = "";
    return;
  }
  const j = await fetch(`/typeahead?q=${encodeURIComponent(q)}&n=8`).then(r=>r.json());
  const items = j.suggestions || [];
  if (!items.length) {
    $("typeahead").classList.add("hidden");
    $("typeahead").innerHTML = "";
    return;
  }
  $("typeahead").classList.remove("hidden");
  $("typeahead").innerHTML = items.map(t => `<div class="taItem">${escapeHtml(t)}</div>`).join("");
  [...$("typeahead").querySelectorAll(".taItem")].forEach((el) => {
    el.onclick = async () => {
      $("searchInput").value = el.textContent;
      $("typeahead").classList.add("hidden");
      await doSearch(el.textContent, true);
    };
  });
}

function escapeHtml(s) {
  return (s||"").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}

async function doSearch(q, isUserSearch=false) {
  currentQuery = q;
  rememberQuery(q);

  // IMPORTANT: MovieLens doesn’t know “Tamil”, “K-drama”, etc. Unless you add metadata.
  // Minimal fix: simple rewrite for demo so it doesn’t look broken.
  let query = q;
  const ql = q.toLowerCase();
  if (ql.includes("tamil")) query = q.replace(/tamil/ig, "romantic drama");

  const payload = {
    query,
    method: "hybrid_ltr",
    k: 12,
    candidate_k: 200,
    rerank_k: 50,
    alpha: 0.5,
    debug: false
  };

  const res = await fetch("/search", {
    method:"POST",
    headers:{ "content-type":"application/json" },
    body: JSON.stringify(payload)
  });
  const j = await res.json();
  const hits = j.hits || [];

  // Hero = top hit
  if (hits.length) {
    setHeroFromHit(hits[0], query);
  } else {
    $("heroTitle").textContent = q;
    $("heroDesc").textContent = "No results. (If you want ‘Tamil’/country/language, you must add that metadata.)";
  }

  // Build “Search results” row (top)
  const rowsEl = $("rows");
  if (isUserSearch) {
    // remove existing first row if it was a previous search row
    const prev = document.querySelector('[data-row="search"]');
    if (prev) prev.remove();
    const row = renderRow(`Search results for “${q}”`, hits, { rowKey: "search" });
    rowsEl.prepend(row);
  }
}

function setHeroFromHit(hit, q) {
  $("heroTitle").textContent = hit.title || "(no title)";
  const meta = parseMeta(hit.text);
  $("heroDesc").textContent =
    `Query: “${q}”. Genres: ${meta.genres}. Tags: ${meta.tags}. Click “Why this?” for a grounded explanation.`;
  $("hero").style.background = `${hashColor(hit.doc_id||"x")}`;
  currentSelected = hit;
}

function renderRow(title, hits, opts={}) {
  const row = document.createElement("section");
  row.className = "row";
  if (opts.rowKey) row.dataset.row = opts.rowKey;

  const h = document.createElement("h2");
  h.className = "rowTitle";
  h.textContent = title;

  const items = document.createElement("div");
  items.className = "rowItems";

  for (const hit of hits) {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.style.background = hashColor(hit.doc_id||hit.title||"x");

    const meta = parseMeta(hit.text);
    tile.innerHTML = `
      <div class="tileMeta">
        <span class="tag">${escapeHtml((meta.genres||"—").split(",")[0] || "genre")}</span>
        <span class="tag">score ${Number(hit.score).toFixed(2)}</span>
      </div>
      <div class="tileTitle">${escapeHtml(hit.title||"(no title)")}</div>
    `;
    tile.onclick = () => openModal(hit);
    items.appendChild(tile);
  }

  row.appendChild(h);
  row.appendChild(items);
  return row;
}

async function loadHomeRows() {
  const rowsEl = $("rows");
  rowsEl.innerHTML = "";

  for (const r of ROWS) {
    const payload = {
      query: r.query,
      method: "hybrid_ltr",
      k: 12,
      candidate_k: 200,
      rerank_k: 50,
      alpha: 0.5,
      debug: false
    };

    const res = await fetch("/search", {
      method:"POST",
      headers:{ "content-type":"application/json" },
      body: JSON.stringify(payload)
    });
    const j = await res.json();
    const hits = j.hits || [];
    rowsEl.appendChild(renderRow(r.title, hits));
  }

  // set hero from the first row’s first result
  const firstTile = rowsEl.querySelector(".tile");
  if (firstTile) {
    // find the first row's first hit by re-running quick search once (cheap enough for demo)
    const res = await fetch("/search", {
      method:"POST",
      headers:{ "content-type":"application/json" },
      body: JSON.stringify({ query: ROWS[0].query, method:"hybrid_ltr", k:1, candidate_k:200, rerank_k:50, alpha:0.5, debug:false })
    });
    const j = await res.json();
    if (j.hits && j.hits[0]) setHeroFromHit(j.hits[0], ROWS[0].query);
  }
}

function openModal(hit) {
  currentSelected = hit;
  const meta = parseMeta(hit.text);

  $("modalTitle").textContent = hit.title || "(no title)";
  $("modalPoster").style.background = hashColor(hit.doc_id||hit.title||"x");
  $("modalGenres").textContent = `Genres: ${meta.genres}`;
  $("modalTags").textContent = `Tags: ${meta.tags}`;
  $("modalMatch").textContent = `score ${Number(hit.score).toFixed(2)}`;

  $("explainText").textContent = "—";
  $("sourcesList").innerHTML = "";
  $("rawBox").textContent = "—";
  $("explainTimings").textContent = "—";

  $("modal").classList.remove("hidden");
}

function closeModal() {
  $("modal").classList.add("hidden");
  stopSpeak();
}

async function groundedWhy(agent=false) {
  if (!currentSelected) return;

  // Make the LLM do an actual Netflix-relevant job:
  // explain why this title matches the user’s query using retrieved metadata
  const q = currentQuery || "gritty crime thriller";
  const promptQuery = `Why is "${currentSelected.title}" a good match for the query "${q}"? Use only retrieved metadata and cite sources.`;

  const payload = {
    query: promptQuery,
    method: "hybrid_ltr",
    k: 8,
    candidate_k: 200,
    rerank_k: 50,
    alpha: 0.5,
    context_k: 6,
    debug: true,
    temperature: 0.2
  };

  const endpoint = agent ? "/agent_answer" : "/answer";
  const res = await fetch(endpoint, {
    method:"POST",
    headers:{ "content-type":"application/json" },
    body: JSON.stringify(payload)
  });
  const j = await res.json();

  $("explainText").textContent = j.answer || "—";
  lastSpoken = j.answer || "";

  const t = j.timings_ms || {};
  const parts = [];
  for (const [k,v] of Object.entries(t)) parts.push(`${k}:${Number(v).toFixed(0)}ms`);
  $("explainTimings").textContent = parts.join(" | ") || "—";

  // sources
  const srcs = j.sources || [];
  $("sourcesList").innerHTML = srcs.map((s, i) => {
    const snippet = (s.snippet || s.text || "").toString();
    return `
      <div class="src">
        <div class="t">[${i+1}] ${escapeHtml(s.title||"(no title)")} <span class="mono small">doc_id=${escapeHtml(s.doc_id||"")}</span></div>
        <div class="small">${escapeHtml(snippet)}</div>
      </div>
    `;
  }).join("");

  $("rawBox").textContent = JSON.stringify(j, null, 2);
}

function rememberQuery(q) {
  if (!currentUser) return;
  const key = `recent:${currentUser}`;
  const prev = JSON.parse(localStorage.getItem(key) || "[]");
  const next = [q, ...prev.filter(x=>x!==q)].slice(0, 12);
  localStorage.setItem(key, JSON.stringify(next));
}

function getRecentQueries() {
  if (!currentUser) return [];
  const key = `recent:${currentUser}`;
  return JSON.parse(localStorage.getItem(key) || "[]");
}

let utter = null;
function speak() {
  if (!lastSpoken) return;
  if (!("speechSynthesis" in window)) {
    alert("Browser speech not available.");
    return;
  }
  stopSpeak();
  utter = new SpeechSynthesisUtterance(lastSpoken);
  utter.rate = 1.0;
  utter.pitch = 1.0;
  window.speechSynthesis.speak(utter);
}
function stopSpeak() {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
}

function setProfile(user) {
  currentUser = user;
  $("profileName").textContent = user.charAt(0).toUpperCase() + user.slice(1);
  $("profileBadge").textContent = user.charAt(0).toUpperCase();
  $("profileBadge").style.background = (user==="gilbert") ? "#FF6B6B" : "#4169E1";
}

function showScreen(idToShow) {
  ["loginScreen","profileScreen","app"].forEach(id => $(id).classList.add("hidden"));
  $(idToShow).classList.remove("hidden");
}

function wireUI() {
  $("loginForm").addEventListener("submit", (e) => {
    e.preventDefault();
    showScreen("profileScreen");
  });

  document.querySelectorAll(".profileCard").forEach(btn => {
    btn.onclick = async () => {
      const user = btn.dataset.user;
      setProfile(user);
      showScreen("app");
      await apiHealth();
      await apiLift();
      await apiLatency();
      await loadHomeRows();

      // preload last query if exists
      const rec = getRecentQueries();
      if (rec.length) {
        $("searchInput").value = rec[0];
        currentQuery = rec[0];
      }
    };
  });

  $("switchProfile").onclick = () => {
    if (confirm("Switch profiles?")) {
      showScreen("profileScreen");
    }
  };

  // typeahead
  let taTimer = null;
  $("searchInput").addEventListener("input", (e) => {
    const q = e.target.value || "";
    clearTimeout(taTimer);
    taTimer = setTimeout(() => typeahead(q), 120);
  });

  $("searchInput").addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
      $("typeahead").classList.add("hidden");
      await doSearch($("searchInput").value, true);
    }
  });

  $("heroPlay").onclick = () => {
    if (currentSelected) openModal(currentSelected);
  };
  $("heroInfo").onclick = () => {
    if (currentSelected) openModal(currentSelected);
  };
  $("heroWhy").onclick = async () => {
    if (currentSelected) {
      openModal(currentSelected);
      await groundedWhy(false);
    }
  };

  $("modalClose").onclick = closeModal;
  $("modalBackdrop").onclick = closeModal;

  $("btnGroundedWhy").onclick = () => groundedWhy(false);
  $("btnAgentWhy").onclick = () => groundedWhy(true);
  $("btnSpeak").onclick = speak;
  $("btnStop").onclick = stopSpeak;

  $("btnSearchLikeThis").onclick = async () => {
    if (!currentSelected) return;
    const title = currentSelected.title || "";
    $("modal").classList.add("hidden");
    $("searchInput").value = title;
    await doSearch(title, true);
  };

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });
}

wireUI();
