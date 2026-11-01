<div align="center">

# StreamLens

### Search, ranking, and discovery for streaming catalogs.

[**Explore the interactive tour ↗**](https://raw.githubusercontent.com/AKilalours/streaming-canvas-search-ltr/main/docs/demos/README_INTERACTIVE.html) · [**Open the code**](https://github.com/AKilalours/streaming-canvas-search-ltr/tree/main/src) · [**See evaluation evidence**](docs/evaluation/DEFINITIVE_NUMBERS.md)

**BM25 + dense retrieval** &nbsp;·&nbsp; **LambdaRank** &nbsp;·&nbsp; **personalized feedback** &nbsp;·&nbsp; **multimodal experiments**

[![CI](https://github.com/AKilalours/streaming-canvas-search-ltr/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AKilalours/streaming-canvas-search-ltr/actions/workflows/ci.yml)

</div>

---

## Generated poster gallery

<p align="center">
  <img src="artifacts/sd_posters/dalle3_pulp_fiction.png" width="190" alt="Generated neo-noir poster artwork from the StreamLens poster demo" />
  &nbsp;&nbsp;
  <img src="artifacts/sd_posters/dalle3_toy_story.png" width="190" alt="Generated animated science-fiction poster artwork from the StreamLens poster demo" />
  &nbsp;&nbsp;
  <img src="artifacts/sd_posters/dalle3_15d3295d.png" width="190" alt="Generated poster artwork from the StreamLens demo" />
</p>

<p align="center"><sub>Generated demo artwork in the repository · these are poster outputs, not screenshots of the app.</sub></p>

## Try the interactive app

```bash
make setup
make serve
```

Then open **[Search UI](http://localhost:8000/demo)** · **[SQL explorer](http://localhost:8000/sql)** · **[Poster demo](http://localhost:8000/diffusion)** · **[API](http://localhost:8000/docs)**

The first run may download model weights. Some generative features require API credentials. See [`env.example`](env.example) and [`docker-compose.example.yml`](docker-compose.example.yml) before enabling integrations.

## A measured result

**nDCG@10: 0.9491** on 150 MovieLens test queries (9,742 titles). The reference result is **0.7506**, from a different candidate width and embedding setup; see the [full evaluation notes](docs/evaluation/DEFINITIVE_NUMBERS.md) for comparison limits and additional datasets.

**p99: 165 ms** in a local `/search` benchmark at concurrency 20 (200 requests). This is a local measurement, not a production SLO.

## What’s inside

| Explore | Open |
|---|---|
| Interactive system walkthrough | [Open tour](https://raw.githubusercontent.com/AKilalours/streaming-canvas-search-ltr/main/docs/demos/README_INTERACTIVE.html) |
| API and search service | [`src/app`](src/app) · [`API docs`](http://localhost:8000/docs) |
| Retrieval and learning to rank | [`src/retrieval`](src/retrieval) · [`src/ranking`](src/ranking) |
| Feedback and contextual bandit | [`src/exploration`](src/exploration) · [`src/causal`](src/causal) |
| Evaluation reports | [`reports/`](reports) · [`definitive numbers`](docs/evaluation/DEFINITIVE_NUMBERS.md) |
| Pipelines and deployment | [`flows/`](flows) · [`docker/`](docker) · [`k8s/`](k8s) |
| Script index | [`scripts/README.md`](scripts/README.md) |

> The feedback learner is a contextual bandit. Online long-term reward and a complete propensity-safe off-policy evaluation loop are not established. See the [evaluation notes](docs/evaluation/DEFINITIVE_NUMBERS.md).

<div align="center">

[**Explore the tour ↗**](https://raw.githubusercontent.com/AKilalours/streaming-canvas-search-ltr/main/docs/demos/README_INTERACTIVE.html) · [Issues](https://github.com/AKilalours/streaming-canvas-search-ltr/issues) · [MIT License](LICENSE)

</div>
