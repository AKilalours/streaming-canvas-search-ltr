<div align="center">

# StreamLens

**Search · rank · discover**

[**Open the interactive showcase ↗**](https://akilalours.github.io/streaming-canvas-search-ltr/) · [Run locally](#run-the-app) · [Evaluation](docs/evaluation/DEFINITIVE_NUMBERS.md)

[![CI](https://github.com/AKilalours/streaming-canvas-search-ltr/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AKilalours/streaming-canvas-search-ltr/actions/workflows/ci.yml)

</div>

---

## Watch the search flow

<a href="docs/site/assets/search-flow.mp4"><img src="docs/site/assets/search-flow.gif" alt="Animated replay of a real StreamLens search response, showing BM25 and dense retrieval feeding a ranked slate" width="100%"></a>

<sub>Click the animation for the MP4. This replay uses a response captured from the running local API; it is not a browser screen recording. <a href="docs/site/data/search-flow.json">View the exact response.</a></sub>

## The app UI

<a href="docs/site/assets/ui-tour.mp4"><img src="docs/site/assets/ui-tour.gif" alt="App-only StreamLens recording, trimmed to remove desktop windows and long pauses" width="100%" /></a>

<sub><a href="docs/site/assets/ui-tour.mp4">Watch the 75-second app-only recording ↗</a> · Captured from the real app, cropped to its window, trimmed, and muted. The original recordings were left untouched.</sub>

<table>
  <tr>
    <td><a href="docs/site/assets/ui-profile.webp"><img src="docs/site/assets/ui-profile.webp" alt="StreamLens profile picker" width="100%" /></a><b>1 · Choose a profile</b></td>
    <td><a href="docs/site/assets/ui-home.webp"><img src="docs/site/assets/ui-home.webp" alt="StreamLens discovery home" width="100%" /></a><b>2 · Discover titles</b></td>
  </tr>
  <tr>
    <td><a href="docs/site/assets/ui-detail.jpg"><img src="docs/site/assets/ui-detail.jpg" alt="Title detail view" width="100%" /></a><b>3 · Open title details</b></td>
    <td><a href="docs/site/assets/ui-ai-stack.jpg"><img src="docs/site/assets/ui-ai-stack.jpg" alt="AI Stack view" width="100%" /></a><b>4 · Inspect AI Stack</b></td>
  </tr>
  <tr>
    <td><a href="docs/site/assets/ui-live.jpg"><img src="docs/site/assets/ui-live.jpg" alt="Live streaming controls" width="100%" /></a><b>5 · Live controls</b></td>
    <td><a href="docs/site/assets/ui-metrics.jpg"><img src="docs/site/assets/ui-metrics.jpg" alt="Metrics view" width="100%" /></a><b>6 · Metrics</b></td>
  </tr>
  <tr>
    <td><a href="docs/site/assets/ui-sql.jpg"><img src="docs/site/assets/ui-sql.jpg" alt="SQL Explorer with labeled sample fixtures" width="100%" /></a><b>7 · SQL Explorer</b></td>
    <td><a href="docs/site/assets/ui-diffusion.jpg"><img src="docs/site/assets/ui-diffusion.jpg" alt="Diffusion poster lab" width="100%" /></a><b>8 · Poster lab</b></td>
  </tr>
  <tr>
    <td><a href="docs/site/assets/ui-series.jpg"><img src="docs/site/assets/ui-series.jpg" alt="Series browse page" width="100%" /></a><b>9 · Browse series</b></td>
    <td><a href="docs/site/assets/ui-my-list.jpg"><img src="docs/site/assets/ui-my-list.jpg" alt="My List empty state" width="100%" /></a><b>10 · My List</b></td>
  </tr>
</table>

**In the UI:** Pick a demo profile, browse or search, open title details, and explore AI Stack, Live, Metrics, SQL Explorer, Diffusion, Series, and My List. The recording shows the UI as captured; it does not imply every control was exercised. Metrics has no saved run in this capture, SQL displays explicitly labeled sample fixtures, and My List is empty.

**Languages:** The profile menu offers 42 choices for generated feed, explanation, poster-description, and text-to-speech requests. Navigation labels remain English. [Language source](src/app/main.py) · [UI selector](src/app/demo_ui/index.html).

<details>
<summary>Show all 42 language choices</summary>

Arabic · Basque · Bengali · Cantonese · Catalan · Croatian · Czech · Danish · Dutch · English · Filipino · Finnish · French (including Canadian) · Galician · German · Greek · Hebrew · Hindi · Hungarian · Icelandic · Indonesian · Italian · Japanese · Korean · Malay · Malayalam · Mandarin · Marathi · Norwegian (Bokmål) · Polish · Portuguese (including Brazilian) · Romanian · Russian · Spanish (including Latin American and Castilian) · Swahili · Swedish · Tamil · Telugu · Thai · Turkish · Ukrainian · Vietnamese

</details>

## Generated image outputs

<p align="center">
  <img src="docs/site/assets/poster-noir.webp" width="170" alt="Generated noir poster artwork" /> &nbsp;
  <img src="docs/site/assets/poster-toon.webp" width="170" alt="Generated animated poster artwork" /> &nbsp;
  <img src="docs/site/assets/poster-drama.webp" width="170" alt="Generated cinematic poster artwork" />
</p>

## Run the app

```bash
make setup
make serve
```

Open [Search](http://localhost:8000/demo) · [SQL explorer](http://localhost:8000/sql) · [Poster lab](http://localhost:8000/diffusion) · [API](http://localhost:8000/docs)

## Measured

**nDCG@10 0.9491** · MovieLens, 150 test queries · **p99 165.32 ms** · local `/search`, concurrency 20. [Conditions and comparison limits](docs/evaluation/DEFINITIVE_NUMBERS.md).

[Retrieval + ranking](src/retrieval) · [Personalization](src/exploration) · [Evaluation reports](reports) · [Scripts index](scripts/README.md)

<sub>The learner is a contextual bandit. Validated long-term watch outcomes and a complete online propensity-safe OPE loop are not established.</sub>
