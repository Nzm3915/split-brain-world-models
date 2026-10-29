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
python prototype/benchmark.py prototype/results              # all five benchmark numbers for every saved model
python prototype/bridge.py prototype/results                 # intervention-calibrated bridge, against callables
python prototype/stitch_frozen.py prototype/results/stitch 3000   # the same idea on two frozen autoencoders
python prototype/stitch_mismatch.py prototype/results/stitch 3000 # conv encoder -> MLP decoder, different latent sizes
python -I prototype/stitch_pretrained.py prototype/results/stitch DATA_DIR 4000 2000   # public ImageNet-pretrained pairs
python prototype/stitch_figure.py prototype/results/stitch 3000   # figure: what each calibration produces
python prototype/export_demo.py prototype/results demo/frames.json 6      # frames for the demo page
python prototype/toy_overlap.py prototype/results/overlap 3000 2   # H7 in a world with common input (overlapping midline strip)
python prototype/run_seeds.py prototype/results/seeds 3000        # replication over training seeds
```

| Question | Prototype result |
| --- | --- |
| Does the link width matter? | Midline-entry prediction error falls with link rank and saturates near d = 4 |
| Unity at the exit (committee readout, no link) | Detection (T1) at intact level; cross-midline same/different (T2) at chance. A shared readout solves T2 even without a link — a unified exit is not evidence of internal integration |
| Do the mouse estimators find the true channel? | The silencing-derived (receiving) direction recovers 73–98% of the true channel at every width and in every training seed; the correlation-derived direction 1–52%. The gap is widest for a thin link, where neither estimator is contaminated by the receiver's own dynamics |
| Intervention-calibrated bridge (pillar 2) | Written against three callables, so it can be pointed at any frozen pair. Self-test on the toy brain: agreement with the true link matrix 0.46 for the intervention-calibrated bridge vs 0.016 for the correlation-calibrated one (ceiling for a rank-4 bridge: 0.86) |
| Stitching two frozen autoencoders | Encoder A joined to decoder B by a thin linear bridge, both modules frozen. With a clean recording, correlation-fitted stitching is slightly ahead (0.0187 vs 0.0202 reconstruction MSE). As the recording is contaminated by shared variance the bridge cannot see, the intervention-fitted bridge does not move (0.0202–0.0206) while correlation degrades by 66% and Procrustes by more. The prototype's prediction transfers outside the brain setting |
| Stitching two **public pretrained** networks (pillar 2b) | ResNet18's trunk wired into ResNet34's ImageNet classifier, both frozen, scored as top-1 agreement with ResNet18's own verdict (floor 0.07). A bridge fitted with **no paired recording at all** — each module's own perturbation response plus the sender's own latent covariance, with no image ever passing through both networks — reaches 0.750, against 0.459 for a correlational bridge fitted on 4000 paired latents with its ridge tuned. Same in all four directions (0.761/0.413/0.360 against 0.489/0.244/0.325). Two things decide it: the rank truncation must be taken in the metric of the sender's own latent covariance (the naive interventional truncation collapses to 0.022 at rank 8), and the two output spaces must be correctly matched — permute B's classes and it falls to 0.003, below the floor. Intervention supplies the output metric, data supplies the input metric, and a bridge needs both |
| Where the toy's prediction stops | The toy's claim that shared nuisance in the recording breaks a correlational bridge does NOT transfer to these 512-dimensional pretrained latents (0.459 to 0.458 as the contamination goes 0 to 4). It needs the nuisance to be a sizeable share of the recorded variance, which 8 directions in 512 are not. The toy result stands; its range is now measured |
| Stitching across an architecture mismatch | Conv encoder (latent 12) to MLP decoder (latent 20): the same law holds — intervention flat at 0.0347 while correlation goes 0.0229 to 0.0409 as the recording is contaminated. What moves is the crossing point: with matched modules it is at contamination 0.5, with mismatched ones at 1–2, because fitting a single matrix to the measured responses is tighter when the two sides differ |
| Replication over training seeds | Four seeds, d = 0/2/16: every direction holds — committee T2 at chance (0.47–0.51) while a shared readout reaches 0.67–0.75 without a link, and the receiving direction beats the correlated one on the ground-truth channel in all of them (clearest at d = 2: 0.95–0.98 vs 0.16–0.52) |
| Calibrating a BMI that replaces the cut link | Plain cross-correlation makes the loop diverge. With a hidden common drive in the recording, correlation-calibrated BMIs restore little integration; intervention-calibrated BMIs keep most of it |
| Connecting a separately trained machine hemisphere (H8) | Unit-to-unit pairing by tuning similarity fails (consistency 0.02 vs 0.47 intact); population alignment works (0.25); closed-loop search over all link weights fails (0.02); alignment first, then closed-loop tuning of 4 gains reaches 0.37, near the backprop optimum (0.39) |
| Dream consistency with common input (H7) | In a world where each hemisphere also sees 2 pixels across the midline, the link adds nothing while awake (−0.06, −0.03) and everything in the open-loop dream (+0.14, +0.13; two seeds): shared input alone makes two unconnected models look like one world. Measure integration with the input cut |
| Recovery after silencing one hemisphere (T7) | 1-2 steps after 4 steps of complete silence; cutting the link slows it by one step and leaves 1.4x the normal error instead of 1.2x, so the link adds to recovery rather than writing the state back (as in the mouse data). A wider link also means a larger loss for the partner when it is cut |
| What does a prediction-trained link carry? (H13) | Only what the receiver's own future needs: even at d = 16 the committee T2 stays at chance. Adding an objective to name what is in the *other* half-field raises committee T2 from 0.51 to 0.99 through the same 4-dim link — and leaves it at 0.48 when there is no link. Integration needs both a channel and an objective, and it competes with prediction for the link's capacity (entry ratio 1.57 → 2.05) |

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
