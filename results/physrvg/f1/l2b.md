# L2B — physrvg on f1

n_groups = 64 (grouped 5-fold x 4 seeds cross-fitting (spec §5: allowed below 100 groups); ~13 test groups per fold); variants searched = 88 (layers x 2 pools); layer selected on half the groups, AUC on the other half (100 evaluations); CI: group bootstrap of the selected cell's out-of-fold AUC. a probe can separate twins by how much the objects move, so read each AUC against the legal-vs-legal reference row (two legal outcomes, different motion), the ROI mask floor, the kinematic baseline and the speed-matched stratum. Energy is not scored here (spec §5.0.3 R5; see L2-rep).

Kinematic baseline: the same probe protocol on the true post-event kinematics (speed_A_post, speed_B_post, ke_ratio, dv_A, dv_B, v_cm_post); the model detects the violation beyond speed only if the paired group-bootstrap CI of model − kinematic AUC (selected cell) is above 0. Speed-matched: groups whose violator's KE is within the tolerance of its legal twin's. Norm routes: ridge to the true Δp (fitted_norm) or to (dv_A, dv_B) composed with true masses (composed_norm), score |Δp̂|; 'conserved' is a threshold on a norm, which a linear logistic probe cannot express.

| contrast | n groups | held-out AUC, overall (selected cell) [CI] | frame pool AUC (layer) | ROI pool AUC (layer) | ROI mask floor | kinematic AUC | ΔAUC vs kinematic | model − kinematic, selected cell [paired CI] | speed-matched: n groups, model / kinematic AUC, diff [CI] | ΔAUC vs mask floor | ROI − mask floor [paired CI] | fitted_norm |Δp̂| AUC (held-out) | composed_norm AUC (held-out) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mom_violate_e015 vs conserve_e015 | 64 | 0.549 (snr+8/L24/frame.post) [0.534, 0.656] | 0.561 (snr+8/L24) | 0.544 (snr+0/L29) | 0.514 | 0.680 | -0.131 | -0.086 [-0.179, +0.012] | 7, 0.633 / 0.520, +0.112 [-0.000, +0.255] | +0.035 | +0.069 [-0.008, +0.145] | 0.489 (snr+0/L27/frame.post) | 0.490 (snr-8/L27/frame.post) | not detected beyond the kinematic baseline |
| mom_violate_e1 vs conserve_e1 | 64 | 0.546 (snr-20/L6/frame.post) [0.542, 0.640] | 0.553 (snr-20/L6) | 0.551 (snr-20/L21) | 0.585 | 0.634 | -0.088 | -0.044 [-0.115, +0.032] | 6, 0.479 / 0.472, +0.007 [-0.382, +0.326] | -0.039 | +0.003 [-0.061, +0.067] | 0.517 (snr-20/L0/roi_pair.post) | 0.495 (snr+8/L27/frame.post) | not detected beyond the kinematic baseline |
| conserve_e1 vs conserve_e015 (legal-vs-legal reference) | 64 | 0.972 (snr-8/L12/roi_pair.post) [0.949, 0.990] | 0.853 (snr-8/L6) | 0.972 (snr-8/L12) | 0.602 | 0.996 | -0.025 | -0.024 [-0.047, -0.006] |  | +0.370 | +0.370 [+0.315, +0.414] | 0.592 (snr-8/L3/frame.post) | 0.523 (snr-20/L27/frame.post) | not detected beyond the kinematic baseline |

### Transfer (train on one contrast, score held-out groups of the other)

Compare with the kinematic transfer AUC: F1 violators are slower in both contrasts, so a speed shortcut can transfer too; only model transfer above the kinematic baseline is evidence.

| train -> test | model cell (selected on train) | model transfer AUC | kinematic transfer AUC | test within-contrast AUC | fitted_norm transfer AUC (within) | composed_norm transfer AUC (within) |
|---|---|---|---|---|---|---|
| mom_violate_e1 vs conserve_e1 -> mom_violate_e015 vs conserve_e015 | snr-20/L6/frame.post | 0.551 ± 0.014 | 0.614 | 0.549 | 0.523 (0.489) | 0.487 (0.490) |
| mom_violate_e015 vs conserve_e015 -> mom_violate_e1 vs conserve_e1 | snr+8/L24/frame.post | 0.537 ± 0.009 | 0.665 | 0.546 | 0.502 (0.517) | 0.490 (0.495) |
