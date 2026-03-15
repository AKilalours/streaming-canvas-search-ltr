const $ = (id) => document.getElementById(id);

let currentProfile = "chrisen";
let lastSpoken = "";

function pctForRank(i, n){
  // Looks like Netflix match: top cards high, tail lower
  const p = Math.max(0, (n - i) / n);
  return Math.round(72 + p * 27); // 72–99
}

async function apiJSON(url, opts){
  const r = await fetch(url, opts);
  return await r.json();
}

async function refreshMeta(){
  try{
    const h = await apiJSON("/health");
    $("api").textContent = h.ready ? "API: ready" : "API: not-ready";
  }catch(e){
    $("api").textContent = "API: down";
  }

  try{
    const lift = await apiJSON("/metrics/lift");
    if(lift.ok){
      $("lift").textContent = `Lift: +${lift.lift.toFixed(4)} nDCG@10`;
    } else {
      $("lift").textContent = "Lift: —";
    }
  }catch(e){
    $("lift").textContent = "Lift: —";
  }

  try{
    const langs = await apiJSON("/languages");
    const sel = $("lang");
    sel.innerHTML = "";
    for(const L of langs.languages){
      const opt = document.createElement("option");
      opt.value = L;
      opt.textContent = L;
      sel.appendChild(opt);
    }
    sel.value = "English";
  }catch(e){}
}

async function typeahead(q){
  if(!q || q.trim().length < 2){
    $("typeahead").style.display = "none";
    $("typeahead").innerHTML = "";
    return;
  }
  const j = await apiJSON(`/suggest?q=${encodeURIComponent(q)}&n=8&profile=${encodeURIComponent(currentProfile)}`);
  const items = j.suggestions || [];
  const root = $("typeahead");
  root.innerHTML = "";
  for(const s of items){
    const d = document.createElement("div");
    d.className = "item";
    d.textContent = s;
    d.onclick = () => {
      $("q").value = s;
      root.style.display = "none";
      runSearchAndRows();
    };
    root.appendChild(d);
  }
  root.style.display = items.length ? "block" : "none";
}

async function searchOnce(query, k=12){
  return await apiJSON("/search", {
    method:"POST",
    headers:{"content-type":"application/json"},
    body: JSON.stringify({
      query, method:"hybrid_ltr", k,
      candidate_k:200, rerank_k:50, alpha:0.5,
      debug:true
    })
  });
}

function makeCard(hit, matchPct){
  const div = document.createElement("div");
  div.className = "card";
  div.innerHTML = `<div class="match">${matchPct}% Match</div><div class="title">${hit.title || "(no title)"}</div>`;
  div.onclick = () => openModal(hit, matchPct);
  return div;
}

async function buildRow(title, query){
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `<div class="rowTitle">${title}</div><div class="rail"></div>`;
  const rail = row.querySelector(".rail");

  const j = await searchOnce(query, 12);
  const hits = j.hits || [];
  for(let i=0;i<hits.length;i++){
    rail.appendChild(makeCard(hits[i], pctForRank(i+1, hits.length)));
  }
  return row;
}

async function runSearchAndRows(){
  const rows = $("rows");
  rows.innerHTML = "";

  // Profile-specific “rails” (this is how it feels like Netflix home rows)
  const rails = currentProfile === "gilbert"
    ? [
        ["Light Comedy", "light comedy"],
        ["Feel Good Romance", "feel good romance"],
        ["Family & Animation", "family animation"],
        ["Drama Picks", "coming of age drama"],
      ]
    : [
        ["Gritty Crime Thrillers", "gritty crime thriller"],
        ["Mind-bending Sci-Fi", "mind bending sci fi"],
        ["War & History", "war drama"],
        ["Dark Comedy", "dark comedy"],
      ];

  for(const [t,q] of rails){
    rows.appendChild(await buildRow(t,q));
  }

  const hint = `Tip: search → click a title → “Answer (Grounded)” → “Agent Retry” → “Speak (server TTS)”`;
  $("hint").textContent = hint;
}

function openModal(hit, matchPct){
  $("modal").classList.add("show");
  $("mTitle").textContent = hit.title || "(no title)";
  $("mMeta").textContent = `${matchPct}% Match • doc_id=${hit.doc_id} • score=${(hit.score ?? 0).toFixed(4)}`;

  // Clear panels
  $("mWhy").textContent = hit.score_breakdown
    ? `Hybrid score is computed from BM25 + Dense. LTR reranker then reorders the top-${50} candidates.`
    : `This title matched your query strongly in hybrid retrieval; LTR then refined the order.`;

  $("mAns").textContent = "—";
  $("mSrcs").innerHTML = "";
  $("mRaw").textContent = JSON.stringify(hit, null, 2);

  $("mGround").onclick = () => doAnswer("/answer");
  $("mAgent").onclick = () => doAnswer("/agent_answer");

  async function doAnswer(endpoint){
    const q = $("q").value.trim() || (hit.title || "");
    const j = await apiJSON(endpoint, {
      method:"POST",
      headers:{"content-type":"application/json"},
      body: JSON.stringify({
        query: q,
        method:"hybrid_ltr",
        k: 10,
        candidate_k: 200,
        rerank_k: 50,
        alpha: 0.5,
        context_k: 6,
        debug: true,
        temperature: 0.2
      })
    });

    $("mAns").textContent = j.answer || "—";
    lastSpoken = j.answer || "";
    $("mSrcs").innerHTML = "";

    const srcs = j.sources || [];
    for(let i=0;i<srcs.length;i++){
      const s = srcs[i];
      const d = document.createElement("div");
      d.className = "src";
      d.innerHTML = `<div class="t">[${i+1}] ${s.title || "(no title)"}</div><div>${s.snippet || ""}</div>`;
      $("mSrcs").appendChild(d);
    }
    $("mRaw").textContent = JSON.stringify(j, null, 2);
  }
}

function closeModal(){
  $("modal").classList.remove("show");
}

async function speakServer(){
  const lang = $("lang").value || "English";
  const text = lastSpoken || $("q").value.trim();
  if(!text) return;
  const url = `/tts?language=${encodeURIComponent(lang)}&text=${encodeURIComponent(text)}`;
  const audio = new Audio(url);
  await audio.play();
}

$("mClose").onclick = closeModal;
$("modal").addEventListener("click", (e)=>{ if(e.target.id==="modal") closeModal(); });

$("q").addEventListener("input", (e)=> typeahead(e.target.value));
$("q").addEventListener("keydown", (e)=>{ if(e.key==="Enter"){ $("typeahead").style.display="none"; runSearchAndRows(); }});

$("btnRefresh").onclick = runSearchAndRows;
$("btnVoice").onclick = speakServer;

$("btnExplain").onclick = () => {
  alert(
    "What you’re seeing:\n\n" +
    "1) Hybrid Retrieval (BM25 + Dense): gets strong candidates fast.\n" +
    "2) LambdaRank LTR rerank: reorders top candidates using learned relevance signals.\n" +
    "3) Grounded RAG + Agentic retry: answers only from retrieved catalog snippets; retries with more context if unsupported."
  );
};

(async function init(){
  await refreshMeta();
  await runSearchAndRows();
})();
