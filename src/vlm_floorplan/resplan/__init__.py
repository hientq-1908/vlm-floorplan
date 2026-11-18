"""ResPlan: vector floorplans turned into room graphs and rendered images.

ResPlan (https://github.com/m-agour/ResPlan, data CC BY 4.0, code MIT) ships
17,000 residential plans as shapely geometry, with no images.
This package does three jobs with it:

- `fetch` downloads the dataset into the gitignored `data/` directory,
- `convert` turns a plan into a `FloorplanGraph`, and `render` draws the same plan
  as a PNG whose 0-1000 grid matches the graph (both go through `frame.PlanView`),
- `build` writes JSONL records (see `vlm_floorplan.record`) plus the images.

See docs/resplan.md for the design choices and THIRD_PARTY_NOTICES.md for attribution.
"""
