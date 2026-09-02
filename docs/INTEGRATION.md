# OpenGait integration

## Compatibility target

The overlay follows the OpenGait API layout used by the authors' research
snapshot:

- `BaseModel` returns `training_feat`, `visual_summary`, or `inference_feat`;
- models and losses are discovered from files under `modeling/models` and
  `modeling/losses`;
- `SetBlockWrapper`, `PackSequenceWrapper`, `HorizontalPoolingPyramid`,
  `SeparateFCs`, and `SeparateBNNecks` are available in `modeling/modules.py`;
- inputs are a list of modality tensors plus labels, types, views, and `seqL`.

OpenGait evolves independently. If a recent checkout changes these APIs, use
this repository as the method reference and adapt imports/return dictionaries
to that version.

## Automatic overlay

From this repository:

```bash
python scripts/install_into_opengait.py /path/to/OpenGait
python scripts/install_into_opengait.py /path/to/OpenGait --apply
```

The first command is a dry run. The second copies explicit files and appends an
evaluator import. Re-running is idempotent. A conflicting destination stops the
installer unless `--force` is supplied.

## Manual overlay

Copy the files as follows:

| This repository | OpenGait destination |
| --- | --- |
| `opengait/modeling/models/UniCrossGait.py` | `opengait/modeling/models/UniCrossGait.py` |
| `opengait/modeling/losses/unicrossgait.py` | `opengait/modeling/losses/unicrossgait.py` |
| `opengait/modeling/losses/unicrossgait_math.py` | `opengait/modeling/losses/unicrossgait_math.py` |
| `opengait/evaluation/unicrossgait_evaluator.py` | `opengait/evaluation/unicrossgait_evaluator.py` |
| `configs/*.yaml` | `configs/unicrossgait/*.yaml` |

Then add this import at the end of `opengait/evaluation/evaluator.py`:

```python
from .unicrossgait_evaluator import evaluate_unicrossgait  # noqa: E402,F401
```

If your OpenGait model/loss `__init__.py` uses explicit imports instead of
dynamic discovery, also export `UniCrossGait` and the three loss classes there.

## Input contract

Each selected sequence directory must yield two synchronized frame sequences.
The templates use this order:

| Index | Modality | Expected batch shape before model conversion |
| ---: | --- | --- |
| 0 | LiDAR depth image | `[N,S,3,H,W]` (or `[N,S,H,W]` for a 1-channel variant) |
| 1 | Camera silhouette | `[N,S,H,W]` |

The reference backbones expect three depth channels and one silhouette channel.
If your depth representation is one-channel, change `backbone_cfg_3d.in_channel`
to 1. If your preprocessing filenames sort differently, change either
`data_in_use`/file order or the two input indices in `model_cfg`.

Both modalities must have synchronized frame counts for the unmodified
OpenGait collate function. The paper aligns and resizes all inputs to 64×64.

## Checkpoint flow

1. Train the teacher config for 40,000 OpenGait iterations.
2. Set `model_cfg.teacher_checkpoint` in the student config to the resulting
   `.pt` file. The loader accepts the `model` dictionary saved by OpenGait and
   extracts keys under `teacher.`.
3. Train the student config. The teacher is frozen before optimizer creation
   and is kept in evaluation mode.
4. During test-only construction, the model builds only student branches. Set
   `evaluator_cfg.restore_hint` to the stage-2 checkpoint and use non-strict
   restore so stored teacher keys are ignored.

The cleaned teacher architecture differs from some intermediate experimental
snapshots. A legacy teacher checkpoint whose parameter names or fusion MLP do
not match this release should be retrained or converted explicitly; silent
partial loading is avoided by default.

## Dataset protocol

The provided evaluator implements SUSTech1K's `00-nm` opposite-modality gallery
and state-varying probe subsets, excluding identical-view comparisons. FreeGait
file naming and gallery/probe construction depend on its preprocessing release;
adapt the masks rather than assuming the SUSTech1K names.
