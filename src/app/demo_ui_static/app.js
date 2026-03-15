const $ = (id) => document.getElementById(id);

let PROFILE = null;
let LANGUAGE = "English";
let CURRENT_ITEM = null;

function pretty(obj){ try{return JSON.stringify(obj,null,2)}catch(e){return String(obj)} }

function grad(seed){
  const n = (Number(seed) || 0) % 6;
  const gs = [
    "linear-gradient(135deg,#667eea 0%,#764ba2 100%)",
    "linear-gradient(135deg,#f093fb 0%,#f5576c 100%)",
    "linear-gradient(135deg,#4facfe 0%,#00f2fe 100%)",
    "linear-gradient(135deg,#43e97b 0%,#38f9d7 100%)",
    "linear-gradient(135deg,#fa709a 0%,#fee140 100%)",
    "linear-gradient(135deg,#30cfd0 0%,#330867 100%)",
  ];
  return gs[n];
}

async function apiHealth(){
  try{
    const h = await fetch("/health").then(r=>r.json());
    $("apiStatus").innerHTML = h.ready ? '<span class="ok">ready</span>' : '<span class="bad">not-ready</span>';
  }catch(e){
    $("apiStatus").innerHTML = '<span class="bad">down</span>';
  }
}

async function apiLift(){
  try{
    const j = await fetch("/metrics/lift").then(r=>r.json());
    if(j.ok){
      const v = Number(j.lift || 0);
      const sign = v>=0 ? "+" : "";
      $("liftText").innerHTML = `<span class="${v>=0 ? "ok":"bad"}">${sign}${v.toFixed(4)} nDCG@10</span>`;
    }else{
      $("liftText").textContent = "—";
    }
  }catch(e){
    $("liftText").textContent = "—";
  }
}

async function loadLanguages(){
  const sel = $("langSelect");
  sel.innerHTML = "";
  try{
    const j = await fetch("/languages").then(r=>r.json());
    const langs = j.languages || [];
    for(const lang of langs){
      const opt = document.createElement("option");
      opt.value = lang;
      opt.textContent = lang;
      sel.appendChild(opt);
    }
  }catch(e){
    ["English","Japanese","Spanish","Russian","Tamil","Telugu","Malayalam"].forEach(x=>{
      const opt=document.createElement("option");
      opt.value=x; opt.textContent=x;
      sel.appendChild(opt);
    });
  }
  sel.value = LANGUAGE;
  sel.onchange = () => {
    LANGUAGE = sel.value;
    $("heroNote").textContent = `TTS language set to: ${LANGUAGE}`;
  };
}

function rowEl(title, items){
  const row = document.createElement("div");
  row.className = "row";

  const h = document.createElement("div");
  h.className = "row-title";
  h.textContent = title;

  const list = document.createElement("div");
  list.className = "row-items";

  for(const it of items){
    const card = document.createElement("div");
    card.className = "card";
    card.style.background = grad(it.poster_seed || it.doc_id);

    const center = document.createElement("div");
    center.className = "center";
    center.textContent = it.title || "(no title)";

    const bottom = document.createElement("div");
    bottom.className = "bottom";
    bottom.innerHTML = `
      <div class="meta">${it.match || "—"}</div>
      <div class="small">${it.genres || ""}</div>
    `;

    card.onclick = () => openItem(it);
    card.appendChild(center);
    card.appendChild(bottom);
    list.appendChild(card);
  }

  row.appendChild(h);
  row.appendChild(list);
  return row;
}

async function loadFeed(){
  if(!PROFILE) return;
  $("rows").innerHTML = `<div class="small">Loading personalized rows…</div>`;
  try{
    const url = `/feed?profile=${encodeURIComponent(PROFILE)}&language=${encodeURIComponent(LANGUAGE)}&rows=6&k=12`;
    const j = await fetch(url).then(r=>r.json());

    const root = $("rows");
    root.innerHTML = "";

    if(j.warning){
      $("heroNote").textContent = j.warning;
    }else{
      $("heroNote").textContent = "";
    }

    for(const row of (j.rows || [])){
      root.appendChild(rowEl(row.title, row.items || []));
    }
  }catch(e){
    $("rows").innerHTML = `<div class="small">Failed to load feed. Check server logs.</div>`;
  }
}

let searchTimer = null;

async function doSearch(q){
  if(!PROFILE) return;
  const query = (q||"").trim();
  if(query.length < 2){
    await loadFeed();
    return;
  }
  $("rows").innerHTML = `<div class="small">Searching “${query}”…</div>`;

  const body = {
    query,
    method: "hybrid_ltr",
    k: 24,
    candidate_k: 200,
    rerank_k: 50,
    alpha: 0.5,
    debug: true
  };

  const res = await fetch("/search", {
    method:"POST",
    headers: {"content-type":"application/json"},
    body: JSON.stringify(body)
  });

  const j = await res.json();
  const items = (j.hits || []).map((h, idx) => ({
    doc_id: h.doc_id,
    title: h.title,
    text: h.text,
    genres: (h.text || "").includes("Genres:") ? (h.text.split("Genres:")[1] || "").split("|")[0].trim() : "",
    match: `${Math.max(80, 99 - idx)}% Match`,
    poster_seed: h.doc_id
  }));

  const root = $("rows");
  root.innerHTML = "";
  root.appendChild(rowEl(`Search results for “${query}”`, items));
}

function openItem(it){
  CURRENT_ITEM = it;
  $("modalTitle").textContent = it.title || "(no title)";
  $("modalText").textContent = it.text || "—";
  $("modalGenres").textContent = it.genres ? `Genres: ${it.genres}` : "Genres: —";
  $("modalMatch").textContent = it.match || "—";
  $("modalDebug").textContent = "Click “Ask AI” to generate a grounded explanation.";
  $("ttsNote").textContent = "";
  $("ttsAudio").style.display = "none";
  $("modal").classList.add("active");
}

function closeModal(){
  $("modal").classList.remove("active");
  stopAudio();
}

function openHow(){ $("howModal").classList.add("active"); }
function closeHow(){ $("howModal").classList.remove("active"); }

async function askAI(){
  if(!CURRENT_ITEM) return;

  $("modalDebug").textContent = "Running Agentic RAG…";

  const q = `Give a short synopsis for "${CURRENT_ITEM.title}". Then explain why a user with profile "${PROFILE}" might like it. Use only the retrieved metadata.`;

  const body = {
    query: q,
    method: "hybrid_ltr",
    k: 10,
    candidate_k: 200,
    rerank_k: 50,
    alpha: 0.5,
    context_k: 6,
    temperature: 0.2,
    debug: true
  };

  const res = await fetch("/agent_answer", {
    method:"POST",
    headers: {"content-type":"application/json"},
    body: JSON.stringify(body)
  });

  const j = await res.json();

  const ans = j.answer || "—";
  $("modalText").textContent = ans;

  // show citations/sources in debug
  const dbg = {
    timings_ms: j.timings_ms || null,
    warning: j.warning || null,
    sources: (j.sources || []).map((s, i)=>({
      i: i+1,
      doc_id: s.doc_id,
      title: s.title,
      snippet: s.snippet
    })),
    raw: j.raw || null
  };

  $("modalDebug").textContent = pretty(dbg);
}

async function speak(){
  const txt = ($("modalText").textContent || "").trim();
  if(!txt || txt === "—") return;

  $("ttsNote").textContent = "Generating audio…";
  $("ttsAudio").style.display = "none";

  const url = `/tts?lang=${encodeURIComponent(LANGUAGE)}&text=${encodeURIComponent(txt.slice(0, 480))}`;
  const res = await fetch(url);
  if(!res.ok){
    const j = await res.json().catch(()=>null);
    $("ttsNote").textContent = j?.detail || "TTS failed. See server logs.";
    return;
  }

  const blob = await res.blob();
  const audioUrl = URL.createObjectURL(blob);

  const a = $("ttsAudio");
  a.src = audioUrl;
  a.style.display = "block";
  a.play().catch(()=>{});
  $("ttsNote").textContent = `Speaking with server TTS (${LANGUAGE}).`;
}

function stopAudio(){
  const a = $("ttsAudio");
  try{
    a.pause();
    a.currentTime = 0;
  }catch(e){}
  $("ttsNote").textContent = "";
}

function showProfiles(){
  $("app").style.display = "none";
  $("profiles").classList.remove("hidden");
}

function selectProfile(p){
  PROFILE = p;
  $("profiles").classList.add("hidden");
  $("app").style.display = "block";

  $("profileName").textContent = p.charAt(0).toUpperCase() + p.slice(1);
  const icon = $("profileIcon");
  icon.textContent = p.charAt(0).toUpperCase();
  icon.className = `profile-icon ${p}`;

  apiHealth();
  apiLift();
  loadFeed();
}

document.addEventListener("DOMContentLoaded", async () => {
  // login flow
  $("loginForm").addEventListener("submit", (e) => {
    e.preventDefault();
    $("login").classList.add("hidden");
    $("profiles").classList.remove("hidden");
  });

  document.querySelectorAll(".profile-card").forEach(el => {
    el.addEventListener("click", () => selectProfile(el.dataset.profile));
  });

  $("switchProfile").addEventListener("click", () => {
    if(confirm("Switch profile?")){
      showProfiles();
    }
  });

  $("modalClose").addEventListener("click", closeModal);
  $("modal").addEventListener("click", (e) => { if(e.target === $("modal")) closeModal(); });

  $("howClose").addEventListener("click", closeHow);
  $("howModal").addEventListener("click", (e) => { if(e.target === $("howModal")) closeHow(); });

  $("btnExplain").addEventListener("click", askAI);
  $("btnSpeak").addEventListener("click", speak);
  $("btnStopAudio").addEventListener("click", stopAudio);

  $("btnHow").addEventListener("click", openHow);
  $("btnRefresh").addEventListener("click", loadFeed);
  $("btnTry").addEventListener("click", async () => {
    $("searchInput").value = "Tamil love story";
    await doSearch("Tamil love story");
  });

  $("searchInput").addEventListener("input", (e) => {
    const q = e.target.value;
    clearTimeout(searchTimer);
    searchTimer = setTimeout(()=>doSearch(q), 220);
  });

  await loadLanguages();
});
