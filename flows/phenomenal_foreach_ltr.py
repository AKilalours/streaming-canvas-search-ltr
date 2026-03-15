from __future__ import annotations

import json
from pathlib import Path

from metaflow import FlowSpec, Parameter, step, resources


class PhenomenalLTRFlow(FlowSpec):
    """
    Foreach fan-out demo that runs LOCALLY by default.

    - foreach: language fan-out
    - join: map-reduce join
    - writes artifacts/ltr_multilang/manifest.json

    Later, when you have AWS set up, you can run the SAME flow on Batch using:
      python flows/phenomenal_foreach_ltr.py run --with batch
    (no code changes needed)
    """

    languages = Parameter("languages", default="English,Swedish,Filipino", help="Comma-separated")
    out_dir = Parameter("out_dir", default="artifacts/ltr_multilang", help="Output directory")

    @step
    def start(self):
        self.langs = [x.strip() for x in str(self.languages).split(",") if x.strip()]
        if not self.langs:
            raise ValueError("No languages provided")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.next(self.train, foreach="langs")

    @resources(cpu=2, memory=4000)
    @step
    def train(self):
        lang = self.input

        # TODO (real training):
        # Call your real LTR training for this language split.
        # For now: produce a deterministic artifact showing the pipeline works.
        artifact_path = Path(self.out_dir) / f"ltr_{lang}.json"
        artifact = {
            "language": lang,
            "model_path": str(Path(self.out_dir) / f"ltr_{lang}.pkl"),
            "status": "placeholder",
        }
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

        self.model_info = artifact
        self.next(self.join)

    @step
    def join(self, inputs):
        manifest = {"models": [i.model_info for i in inputs]}
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        Path(self.out_dir, "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.manifest = manifest
        self.next(self.end)

    @step
    def end(self):
        print("✅ Wrote:", str(Path(self.out_dir) / "manifest.json"))


if __name__ == "__main__":
    PhenomenalLTRFlow()
