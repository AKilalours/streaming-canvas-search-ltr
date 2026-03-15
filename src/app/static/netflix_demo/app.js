let PROFILE = null;
let LAST_QUERY = "";
let LAST_HITS = [];
let SELECTED = null;

const $ = (id) => document.getElementById(id);

function show(el){ el.classList.remove("hidden"); }
function hide(el){ el.classList.add("hidden"); }

function scoreToMatch(score, minS, maxS){
  if (maxS - minS < 1e-9) return 78;
  const x = (score - minS) / (maxS - minS);
  return Math.max(55, Math.min(99, Math.round(55 + x * 44)));
}

async function apiJSON(url, opts){
  const r = await fetch(url, opts);
  const j = await r.json();
  if (!r.ok) throw new Error(j?.detail || j?.error || r.statusText);
  return j;
}

async function refreshMeta(){
  try{
    const h = await apiJSON("/health");
    $("apiPill").textContent = `API: ${h.ready ? "ready" : "not-ready"}`;
  }catch(e){
    $("apiPill").textContent = "API: down";
  }

  try{
    const lift = await apiJSON("/metrics/lift");
    if (lift.ok){
      $("liftPill").textContent = `Lift: +${lift.lift.toFixed(4)} nDCG@10`;
      $("liftPill").style.color = "#46d369";
    } else {
      $("liftPill").textContent = "Lift: —";
      $("liftPill").style.color = "#fff";
    }
  }catch(e){
    $("liftPill").textContent = "Lift: —";
    $("liftPill").style.color = "#fff";
  }
}

async function loadLanguages(){
  const sel = $("langSelect");
  sel.innerHTML = "";
  try{
    const j = await apiJSON("/languages");
    for (const lang of j.languages){
      const opt = document.createElement("option");
      opt.value = lang;
      opt.textContent = lang;
      sel.appendChild(opt);
    }
    sel.value = "English";
  }catch(e){
    const opt = document.createElement("option");
    opt.value = "English";
    opt.textContent = "English";
    sel.appendChild(opt);
  }
}

async function typeahead(q){
  const box = $("typeahead");
  if (!q || q.trim().length < 2){
    hide(box);
    box.innerHTML = "";
    return;
  }
  try{
    const j = await apiJSON(`/suggest?q=${encodeURIComponent(q)}&n=8&profile=${encodeURIComponent(PROFILE||"chrisen")}`);
    const items = j.suggestions || [];
    if (!items.length){
      hide(box); box.innerHTML = ""; return;
    }
    box.innerHTML = items.map(t => `<div class="opt">${t}</div>`).join("");
    show(box);
    Array.from(box.querySelectorAll(".opt")).forEach((el) => {
      el.onclick = () => {
        $("q").value = el.textContent;
        hide(box);
        doSearch();
      };
    });
  }catch(e){
    hide(box);
    box.innerHTML = "";
  }
}

function renderRow(title, hits){
  const row = document.createElement("section");
  row.className = "row";
  row.innerHTML = `<h2>${title}</h2><div class="rail"></div>`;
  const rail = row.querySelector(".rail");

  if (!hits.length){
    rail.innerHTML = `<div class="muted">No results.</div>`;
    return row;
  }

  const scores = hits.map(h => Number(h.score || 0));
  const minS = Math.min(...scores);
  const maxS = Math.max(...scores);

  hits.forEach((h, idx) => {
    const t = (h.title || "(no title)").trim();
    const match = scoreToMatch(Number(h.score||0), minS, maxS);
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.innerHTML = `
      <div class="match">${match}% Match</div>
      <div class="t">${t}</div>
    `;
    tile.onclick = () => openModal(h, match);
    rail.appendChild(tile);
  });

  return row;
}

async function loadHomeRows(){
  const root = $("rows");
  root.innerHTML = "";

  // Personalize row seeds by profile (simple but effective for demo)
  const baseSeeds = PROFILE === "gilbert"
    ? [
        {name:"Because you like Feel-Good", q:"feel good comedy"},
        {name:"Romance Picks", q:"romantic comedy"},
        {name:"Family & Animation", q:"family animation"},
        {name:"Light Comedy", q:"light comedy"},
      ]
    : [
        {name:"Gritty & Crime", q:"gritty crime thriller"},
        {name:"Sci-Fi & Mind-Bending", q:"mind bending sci fi"},
        {name:"Action & Adventure", q:"action adventure"},
        {name:"Dark Comedy", q:"dark comedy"},
      ];

  // Also mix dynamic suggestions (so rows aren’t stale)
  let dyn = [];
  try{
    const sug = await apiJSON(`/suggest?n=6&profile=${encodeURIComponent(PROFILE||"chrisen")}`);
    dyn = (sug.suggestions || []).slice(0, 4).map((s, i) => ({name:`Suggested: ${s}`, q:s}));
  }catch(e){}

  const rows = [...baseSeeds, ...dyn].slice(0, 6);

  // Load sequentially (less spiky latency than firing 6 searches at once)
  for (const r of rows){
    const hits = await searchAPI(r.q, 18);
    root.appendChild(renderRow(r.name, hits));
  }
}

async function searchAPI(query, k){
  const payload = {
    query,
    method: "hybrid_ltr",
    k,
    candidate_k: 200,
    rerank_k: 50,
    alpha: 0.5,
    debug: true,
  };
  const j = await apiJSON("/search", {
    method:"POST",
    headers: {"content-type":"application/json"},
    body: JSON.stringify(payload),
  });
  return j.hits || [];
}

async function doSearch(){
  const q = $("q").value.trim();
  if (!q) return;
  LAST_QUERY = q;

  const root = $("rows");
  root.innerHTML = "";
  const hits = await searchAPI(q, 30);
  LAST_HITS = hits;
  root.appendChild(renderRow(`Search results for “${q}”`, hits));
}

function openModal(hit, match){
  SELECTED = hit;
  $("mTitle").textContent = (hit.title || "(no title)").trim();
  $("mMatch").textContent = `${match}% Match`;
  $("mDoc").textContent = `doc_id=${hit.doc_id} | score=${Number(hit.score||0).toFixed(4)}`;

  $("mWhy").textContent = "—";
  hide($("mWarn"));
  $("mSources").innerHTML = "";
  $("mRaw").textContent = "—";

  show($("modal"));
}

function closeModal(){
  hide($("modal"));
  SELECTED = null;
}

function renderSources(srcs){
  const root = $("mSources");
  root.innerHTML = "";
  if (!srcs || !srcs.length){
    root.innerHTML = `<div class="muted">No sources returned.</div>`;
    return;
  }
  srcs.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "src";
    div.innerHTML = `
      <div><b>[${i+1}] ${s.title || "(no title)"}</b> <span class="muted mono">doc_id=${s.doc_id || ""}</span></div>
      <div class="muted">${(s.snippet || s.text || "").toString()}</div>
    `;
    root.appendChild(div);
  });
}

async function whyThis(agent=false){
  if (!SELECTED) return;
  const lang = $("langSelect").value || "English";
  const payload = {
    query: LAST_QUERY || $("q").value.trim() || (SELECTED.title || ""),
    doc_id: SELECTED.doc_id,
    method: "hybrid_ltr",
    k: 8,
    candidate_k: 200,
    rerank_k: 50,
    alpha: 0.5,
    context_k: 6,
    debug: true,
    agent: agent,
    language: lang,
  };

  $("mWhy").textContent = "Thinking…";
  hide($("mWarn"));

  const j = await apiJSON("/why", {
    method:"POST",
    headers: {"content-type":"application/json"},
    body: JSON.stringify(payload),
  });

  $("mWhy").textContent = j.answer || "—";
  if (j.warning){
    $("mWarn").textContent = j.warning;
    show($("mWarn"));
  }
  renderSources(j.sources || []);
  $("mRaw").textContent = JSON.stringify(j, null, 2);
}

async function speakServer(){
  const txt = $("mWhy").textContent || "";
  if (!txt || txt === "—" || txt === "Thinking…") return;
  const lang = $("langSelect").value || "English";
  const q = new URLSearchParams({ text: txt, language: lang }).toString();
  const audio = $("ttsPlayer");
  audio.src = `/tts?${q}`;
  await audio.play();
}

function bind(){
  $("loginBtn").onclick = () => { hide($("login")); show($("profiles")); };

  document.querySelectorAll(".profile").forEach(btn => {
    btn.onclick = async () => {
      PROFILE = btn.dataset.profile;
      hide($("profiles"));
      await refreshMeta();
      await loadHomeRows();
    };
  });

  $("switchBtn").onclick = () => {
    show($("profiles"));
  };

  $("refreshBtn").onclick = async () => {
    await refreshMeta();
    await loadHomeRows();
  };

  $("reloadRowsBtn").onclick = async () => {
    await loadHomeRows();
  };

  $("searchBtn").onclick = doSearch;

  $("q").addEventListener("input", (e) => typeahead(e.target.value));
  $("q").addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
  });

  $("closeModal").onclick = closeModal;
  $("modal").addEventListener("click", (e) => {
    if (e.target.id === "modal") closeModal();
  });

  $("whyBtn").onclick = () => whyThis(false);
  $("agentWhyBtn").onclick = () => whyThis(true);
  $("speakBtn").onclick = speakServer;

  $("helpBtn").onclick = () => show($("help"));
  $("openExplainBtn").onclick = () => show($("help"));
  $("closeHelp").onclick = () => hide($("help"));
}

(async function main(){
  bind();
  await loadLanguages();
  await refreshMeta();
})();
