# Report evidence scripts

Scripts that produced the measurements and figures in the final report.
Outputs go to `evidence_out/` (or `$EVIDENCE_DIR`). Run from the repo root with the venv.

| Script | Produces |
|---|---|
| `bench.py evidence_out/bench.json` | End-to-end accuracy and latency on 20 raw images per class (seed 7) |
| `pick.py BYE` | Frames per letter that survive the browser path (YUV420 + JPEG q70) |
| `demo.py BYE` | Drives the running app in Edge with a fake camera; saves screenshots to `img/` |
| `figs.py` | Report diagrams and charts in `fig/` |

The dataset is expected at `archive (5)/data/<class>/<n>.jpg`. `demo.py` needs the server
running on 127.0.0.1:8000 and Microsoft Edge installed.
