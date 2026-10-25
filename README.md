# System 1 decision model: workshop notebook

Prosit Summit, Zakopane, 26-27 Oct 2026.

Open `system1_workshop.ipynb` in Google Colab or Jupyter and run all cells. With `LIVE = False` (the default) it needs no GPU and no model download.

## Contents

- `system1_workshop.ipynb`: the notebook.
- `data/items_en.jsonl`: Tacit v0.2 English dev split, 750 items. CC BY 4.0, https://tacitbench.ai. The canary string asks that the data never be used for training.
- `data/predictions/*.jsonl`: Tacit reference predictions on the same items (probabilities per option).
- `data/basal-2.0-mini.CALIBRATION.json`: temperatures and certified thresholds shipped with `Remek/basal-2.0-mini` (Apache-2.0).
- `data/demo_cache.json`: basal-2.0-mini answers for the notebook's own example e-mails, so the demo cells work offline.
- `system1_workshop.executed.ipynb`: the same notebook with outputs from a live run on an M5 Mac (backup for the demo).
- `build_notebook.py`: regenerates the notebook.

## Rebuilding the demo cache

To regenerate `data/demo_cache.json` after editing the example e-mails, run on a machine with the model:

```bash
S1_LIVE=1 S1_SAVE_CACHE=1 jupyter nbconvert --to notebook --execute system1_workshop.ipynb --output /tmp/run.ipynb
```

## Open in Colab

https://colab.research.google.com/github/agentGreg/system1-workshop/blob/main/system1_workshop.ipynb

## Licences and sources

- **Notebook and code:** MIT.
- **`data/items_en.jsonl` and `data/predictions/`:** from the Tacit benchmark v0.2 (https://github.com/tacit-bench/tacit), CC BY 4.0. Do not use them for training.
- **`data/*.CALIBRATION.json`:** shipped with the basal models by Remigiusz Kinas (https://huggingface.co/Remek), Apache-2.0.
