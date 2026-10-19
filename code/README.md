# Source provenance

`original_snapshot/` contains the six exact source files recorded by the fixed-30-epoch experiments. SHA-256 values in `artifacts/paper_results/protocol.json` verify these original source files.

Only the supported entry points under `scripts/` are documented for this release. Some original modules include historical development entry points that are preserved for provenance; running those directly is not the reproduction workflow for the manuscript.

`fixed_budget_posthoc.py` is the post-training diagnostic script. Its public wrapper sets portable paths without editing the archived computations.
