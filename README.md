# split-brain-world-models

Two world models, each seeing one half of the visual field, joined by a narrow "callosal" link.
What must the link carry — and how much, and calibrated how — for the two halves to render one
consistent world? A computational testbed for Watanabe's subjective test of machine consciousness
(a biological hemisphere connected to a machine hemisphere through a brain–machine interface).

## Documents (Japanese)

- [docs/02_split_brain_world_models.md](docs/02_split_brain_world_models.md) — research plan (rebuilt 2026-10-29):
  three main questions carried over from the mouse project (what to carry, unity at the exit, calibration),
  the split-brain benchmark, hypotheses with falsification criteria, and the results of a CPU prototype.
  Replaces the earlier `research_plan.md` (in the git history).

## Demo

`demo/index.html` plays the trained models frame by frame: the world on top, each hemisphere's prediction below,
with the midline crossing marked. Switch between intact / thin link / cut / correlation-calibrated BMI /
intervention-calibrated BMI, and between wake and the open-loop "dream". Frames are exported by
`prototype/export_demo.py`.

## Prototype (`prototype/`, CPU, minutes per model)

```
python prototype/toy_split_brain.py prototype/results 3000        # train d = 0,1,2,4,16; T1/T2/T5 probes, silencing
python prototype/ground_truth_channel.py prototype/results        # do the mouse estimators recover the true channel?
python prototype/dream_consistency.py prototype/results           # T8: consistency awake vs open-loop "dream"
python prototype/bmi_calibration.py prototype/results 16          # cut the link, replace it by a linear BMI:
DRIVE_AMP=0.15 python prototype/bmi_calibration.py prototype/results 16   #   correlation- vs intervention-calibrated
python prototype/aux_objective.py prototype/results/aux 3000      # H13: does the training objective decide what the link carries?
python prototype/routing.py prototype/results 600                 # H8: connect a separately trained "machine" hemisphere
python prototype/recovery.py prototype/results               # T7: recovery after silencing one hemisphere
python prototype/export_demo.py prototype/results demo/frames.json 6      # frames for the demo page
python prototype/toy_overlap.py prototype/results/overlap 3000 2   # H7 in a world with common input (overlapping midline strip)
python prototype/run_seeds.py prototype/results/seeds 3000        # replication over training seeds
```

| Question | Prototype result |
| --- | --- |
| Does the link width matter? | Midline-entry prediction error falls with link rank and saturates near d = 4 |
| Unity at the exit (committee readout, no link) | Detection (T1) at intact level; cross-midline same/different (T2) at chance. A shared readout solves T2 even without a link — a unified exit is not evidence of internal integration |
| Do the mouse estimators find the true channel? | The silencing-derived (receiving) direction recovers 74–97% of the true channel at every width; the correlation-derived direction 1–52% |
| Calibrating a BMI that replaces the cut link | Plain cross-correlation makes the loop diverge. With a hidden common drive in the recording, correlation-calibrated BMIs restore little integration; intervention-calibrated BMIs keep most of it |
| Connecting a separately trained machine hemisphere (H8) | Unit-to-unit pairing by tuning similarity fails (consistency 0.02 vs 0.47 intact); population alignment works (0.25); closed-loop search over all link weights fails (0.02); alignment first, then closed-loop tuning of 4 gains reaches 0.37, near the backprop optimum (0.39) |
| Recovery after silencing one hemisphere (T7) | 1-2 steps after 4 steps of complete silence; cutting the link slows it by one step and leaves 1.4x the normal error instead of 1.2x, so the link adds to recovery rather than writing the state back (as in the mouse data). A wider link also means a larger loss for the partner when it is cut |
| What does a prediction-trained link carry? | Only what the receiver's own future needs: even at d = 16 the committee T2 stays at chance (H13) |

## Related

- [callosal-dims](https://github.com/Nzm3915/callosal-dims) — the preceding mouse project: which dimensions the corpus callosum
  carries, measured with unilateral silencing in mouse ALM (DANDI 000894). Its measures (receiving vs correlated
  subspace, link dimension, calibration by intervention, recovery time) are reused here.

## Author

**Lan Zhenlin (藍振臨) — [@Nzm3915](https://github.com/Nzm3915)**, Electrical & Electronic Engineering, Korea University.
Designed the research plan: the split-brain benchmark, the conditions that map onto competing views of the split
brain (including the link-free committee condition), and the bridge from the mouse results of callosal-dims to a
testbed with a known ground-truth channel. Prototype code was written with an AI pair-programmer (Claude Code)
under his direction and review.
