# vlm-floorplan

A floorplan assistant built on Qwen3.5-4B.
It turns a captured floorplan image into a room graph: rooms are nodes, and the connections between them are edges.
The user then edits that graph by chat or by dragging.
The output is the graph.
Walls and geometry are never edited.

This is a self-learning, non-commercial project (experiments and research).
The code is written to be read: each module explains the design choice behind it, and `docs/architecture.md` ties them together.

## Status

This repository currently holds milestone M1:

- the graph schema (`schema.py`, `room_types.py`, `compact.py`, `record.py`),
- a CubiCasa5K converter (`converters/`),
- converters for MSD and Swiss Dwellings, graph-only (`converters/msd*.py`, `converters/swiss*.py`),
- fetch and build scripts for CubiCasa5K, MSD and Swiss Dwellings (`data/`),
- a ResPlan converter, renderer and build script (`resplan/`),
- an overlay viewer and an HTML contact sheet (`viewer/`),
- typed graph edit operations, the graph validator and edit history (`edits/`, see `docs/edit-operations.md`),
- tests, including a CPU check of the training format.

It also holds the code for milestone M2, the zero-shot baseline: evaluation metrics, a vLLM serving script and a ClearML job (`evaluation/`, `deploy/vllm/`, see `docs/evaluation.md`).
The GPU run itself has not happened yet.

Planned next: the zero-shot baseline run, perception fine-tuning, the editing agent, the app and retrieval.
See the milestone list in `docs/architecture.md`.

## Quick start

The project uses [uv](https://docs.astral.sh/uv/) for everything.

```bash
uv sync          # core and dev dependencies, no torch
uv run pytest    # the training-format tests skip themselves without the ml group
```

Tests that hit a real server are skipped by default.
Run them with `uv run pytest -m network` (the Zenodo one takes a few minutes).

The heavy ML stack (torch, transformers, TRL, PEFT, ClearML) lives in a separate group.
Only the training-format tests need it.

```bash
uv run --group ml pytest -m hf    # downloads the Qwen3.5-4B processor files (a few MB) once
```

## Fetch, convert and view data

Everything below writes under `data/`, which git ignores.

```bash
# 1. Fetch 20 random CubiCasa5K plans, or pick plans by id.
uv run vlm-floorplan fetch cubicasa --count 20
uv run vlm-floorplan fetch cubicasa --plan colorful/8759 --plan high_quality/1318

# 2. Convert the fetched plans to JSONL (one PlanRecord per line).
uv run vlm-floorplan build cubicasa --out-dir data/processed

# 3. Look at one plan: the graph drawn over the image.
uv run vlm-floorplan view data/processed/cubicasa5k.jsonl --id cubicasa5k/colorful/8759 --out plan.png

# 4. Look at many plans at once: an HTML contact sheet.
uv run vlm-floorplan sheet data/processed/cubicasa5k.jsonl --limit 24 --out-dir data/sheets/cubicasa
```

CubiCasa5K is one 5.5 GB zip on Zenodo.
The fetch script reads single plans with HTTP range requests, so it never downloads the whole archive.
Zenodo is sometimes slow and answers 500 or 504 now and then.
The script retries, and a second run skips files that are already on disk.

Useful `build` options: `--split train|val|test` and `--limit N` (a seeded random subset).
Plans with mostly unlabeled rooms are kept and flagged with `quality.mostly_unlabeled`; training-data assembly filters on that flag.

MSD and Swiss Dwellings are graph-only: they ship vector geometry and no images.
Their records have an empty `image` and feed editing practice and the later retrieval pool.
See `docs/msd-swiss.md` for what is fetched, how links are derived and why.

```bash
# MSD: 50 random plans (about 0.5 MB), or --all (4167 plans, about 40 MB).
uv run vlm-floorplan fetch msd --count 50
uv run vlm-floorplan build msd --out-dir data/processed

# Swiss Dwellings: the first ~100 apartments, or --all (45 000 apartments, 290 MB download).
uv run vlm-floorplan fetch swiss --apartments 100
uv run vlm-floorplan build swiss --out-dir data/processed
```

`build msd` and `build swiss` accept `--limit N` too; they have no `--split`, because neither dataset ships one.

In the overlay, a dot marks each room, colored by type.
A hollow grey dot is a room whose type the dataset does not say (`unknown`).
A green ring marks a room with an exterior door.
Door links are thick orange lines, openings are dashed blue lines, and wall links are thin dotted grey lines.

## ResPlan data

ResPlan (<https://github.com/m-agour/ResPlan>) provides 17,000 vector floorplans.
This project turns them into room graphs and renders them as images.
The data is licensed CC BY 4.0 and its code MIT, so attribution is required: see `THIRD_PARTY_NOTICES.md`.

```bash
uv run python -m vlm_floorplan.resplan.fetch   # about 100 MB into the gitignored data/resplan
uv run python -m vlm_floorplan.resplan.build   # records and images into data/resplan/processed
```

Takedowns: ResPlan removes a plan when a rights holder asks, and then lists its id.
The fetch script checks for that list and tells you when it exists.
Put removed ids in a file and pass it to the builder with `--exclude-ids`.
Design notes and the takedown procedure are in `docs/resplan.md`.

## Project layout

| Path | What it holds |
|---|---|
| `src/vlm_floorplan/schema.py` | `FloorplanGraph`, `Floor`, `Room`, `Link` and their validators. |
| `src/vlm_floorplan/room_types.py` | The coarse room-type mapping for each dataset, including the Finnish label table. |
| `src/vlm_floorplan/compact.py` | The compact JSON text form that the model reads and writes. |
| `src/vlm_floorplan/record.py` | `PlanRecord`: a graph plus polygons, source labels and quality flags. |
| `src/vlm_floorplan/converters/` | CubiCasa5K, MSD and Swiss Dwellings conversion to records, and the code they share (`assemble.py`, `adjacency.py`, `facing.py`; the plan view is `resplan/frame.py`). |
| `src/vlm_floorplan/data/` | Fetch (range requests), build (JSONL) and JSONL helpers. |
| `src/vlm_floorplan/resplan/` | ResPlan fetch, plan-to-graph converter, PNG renderer and build script. |
| `src/vlm_floorplan/viewer/` | The overlay drawing and the contact sheet. |
| `src/vlm_floorplan/edits/` | Typed edit ops, `apply_op`, the graph validator, tool schemas, text summaries and edit history. |
| `src/vlm_floorplan/evaluation/` | The evaluation harness, parser and metrics, and the ClearML baseline launcher (`docs/evaluation.md`). |
| `deploy/vllm/` | The vLLM serving script for the baseline job. |
| `tests/` | Unit tests on synthetic fixtures, plus the training-format tests. |

## Licenses and attribution

The project is non-commercial.
Please keep it that way: do not use the data, anything derived from it, or models trained on it in commercial work.

| Dataset | License | Used for |
|---|---|---|
| [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) | CC BY-NC 4.0 in the repository, CC BY-NC-SA 4.0 on the [Zenodo record](https://zenodo.org/records/2613548). Treat it as non-commercial and share-alike. | Real floorplan images with room polygons and doors. |
| [ResPlan](https://github.com/m-agour/ResPlan) | Data CC BY 4.0, code MIT. It ships vectors only, no images. | Room graphs for editing practice and retrieval, plus rendered images for perception augmentation. |
| [MSD](https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9) | CC BY 4.0 | Graph-only plans of building storeys: editing practice and the retrieval pool. |
| [Swiss Dwellings](https://zenodo.org/records/7788422) | CC BY 4.0 | Graph-only apartments: editing practice and the retrieval pool. |

Attribution for CubiCasa5K: Kalervo, A., Ylioinas, J., Häikiö, M., Karhu, A., Kannala, J., "CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis", Scandinavian Conference on Image Analysis (SCIA), 2019.

Attribution for MSD (CC BY 4.0): Casper van Engelenburg, Seyran Khademi, Fatemeh Mostafavi, Matthias Standfest, Michael Franzen, "Modified Swiss Dwellings", 4TU.ResearchData, 2023, [doi:10.4121/e1d89cb5-6872-48fc-be63-aadd687ee6f9.v2](https://doi.org/10.4121/e1d89cb5-6872-48fc-be63-aadd687ee6f9.v2).

Attribution for Swiss Dwellings (CC BY 4.0): Matthias Standfest, Michael Franzen, Yvonne Schröder, Luis Gonzalez Medina, Yarilo Villanueva Hernandez, Jan Hendrik Buck, Yen-Ling Tan, Milena Niedzwiecka, Rachele Colmegna, "Swiss Dwellings: A large dataset of apartment models including aggregated geolocation-based simulation results covering viewshed, natural light, traffic noise, centrality and geometric analysis", Zenodo, 2022, [doi:10.5281/zenodo.7788422](https://doi.org/10.5281/zenodo.7788422).

The fetch and build scripts write an `ATTRIBUTION.txt` next to the MSD and Swiss Dwellings data they produce, so it travels with any copy of the converted graphs.
These graphs are modified from the originals (converted to room graphs), as CC BY 4.0 asks you to state.

What this means for the repository:

- No dataset images and no derived data are committed.
  Plans converted from CubiCasa5K, and overlays drawn on its images, stay on your machine.
- The tests use synthetic plans written in code (`tests/cubicasa_fixture.py`).
- Model weights trained on CubiCasa5K are non-commercial too.
- ClearML is used for compute only.
  After a job finishes, its outputs are downloaded locally and deleted from ClearML, so nothing from this project stays in a shared workspace.
- The project's own code has no license file yet.
  Until one is added, treat it as all rights reserved.
