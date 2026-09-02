# Configuration templates

The two YAML files mirror the paper's two-stage training procedure. They are
copied to `configs/unicrossgait/` inside OpenGait by the installer.

## Required edits

1. Replace all `YOUR_...` path values.
2. Inspect the sorted files in one preprocessed sequence directory. Set
   `data_in_use` so exactly the synchronized depth and silhouette files are
   loaded, in that order.
3. Change `camera_input_index`/`depth_input_index` if your order differs.
4. Confirm input channels and normalization. The templates expect a 3-channel
   depth representation and a 1-channel silhouette, both stored on a 0–255
   scale.
5. After stage 1, set `teacher_checkpoint` in the student YAML to the teacher
   `.pt` file.
6. Confirm `class_num` against the partition used for training. It is 250 for
   the split described in the paper.

`data_in_use: [true, true]` is intentionally a clean two-file convention, not
a claim about the filename indices in every SUSTech1K preprocessing release.

## Objective switch

The student template enables `use_student_ce` because identity classification
is active in the supplied author training path and supplies the BNNeck logits
used by Distillation Balancing. Set it to `false` to omit the two `softmax_*`
outputs and follow the compact Eq. (28) objective literally. The corresponding
loss entries may remain in `loss_cfg`; OpenGait computes only keys returned in
`training_feat`.
