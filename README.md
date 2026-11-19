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
- a fetch script and a build script for CubiCasa5K (`data/`),
- a ResPlan converter, renderer and build script (`resplan/`),
- an overlay viewer and an HTML contact sheet (`viewer/`),
- tests, including a CPU check of the training format.

Planned next: MSD and Swiss Dwellings converters, a zero-shot baseline, perception fine-tuning, the editing agent, the app and retrieval.
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
| `src/vlm_floorplan/converters/` | CubiCasa5K SVG parsing and conversion to a record. |
| `src/vlm_floorplan/data/` | Fetch (range requests), build (JSONL) and JSONL helpers. |
| `src/vlm_floorplan/resplan/` | ResPlan fetch, plan-to-graph converter, PNG renderer and build script. |
| `src/vlm_floorplan/viewer/` | The overlay drawing and the contact sheet. |
| `tests/` | Unit tests on synthetic fixtures, plus the training-format tests. |

## Licenses and attribution

The project is non-commercial.
Please keep it that way: do not use the data, anything derived from it, or models trained on it in commercial work.

| Dataset | License | Used for |
|---|---|---|
| [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) | CC BY-NC 4.0 in the repository, CC BY-NC-SA 4.0 on the [Zenodo record](https://zenodo.org/records/2613548). Treat it as non-commercial and share-alike. | Real floorplan images with room polygons and doors. |
| [ResPlan](https://github.com/m-agour/ResPlan) | Data CC BY 4.0, code MIT. It ships vectors only, no images. | Room graphs for editing practice and retrieval, plus rendered images for perception augmentation. |
| [MSD](https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9) | CC BY 4.0 | Planned. |
| [Swiss Dwellings](https://zenodo.org/records/7788422) | CC BY 4.0 | Planned. |

Attribution for CubiCasa5K: Kalervo, A., Ylioinas, J., Häikiö, M., Karhu, A., Kannala, J., "CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis", Scandinavian Conference on Image Analysis (SCIA), 2019.

What this means for the repository:

- No dataset images and no derived data are committed.
  Plans converted from CubiCasa5K, and overlays drawn on its images, stay on your machine.
- The tests use synthetic plans written in code (`tests/cubicasa_fixture.py`).
- Model weights trained on CubiCasa5K are non-commercial too.
- ClearML is used for compute only.
  After a job finishes, its outputs are downloaded locally and deleted from ClearML, so nothing from this project stays in a shared workspace.
- The project's own code has no license file yet.
  Until one is added, treat it as all rights reserved.
