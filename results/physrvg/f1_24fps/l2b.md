# L2B — physrvg on f1

n_groups = 64 (grouped 5-fold x 4 seeds cross-fitting (spec §5: allowed below 100 groups); ~13 test groups per fold); variants searched = 88 (layers x 2 pools); layer selected on half the groups, AUC on the other half (100 evaluations); CI: group bootstrap of the selected cell's out-of-fold AUC. a probe can separate twins by how much the objects move, so read each AUC against the legal-vs-legal reference row (two legal outcomes, different motion), the ROI mask floor, the kinematic baseline and the speed-matched stratum. Energy is not scored here (spec §5.0.3 R5; see L2-rep).

Kinematic baseline: the same probe protocol on the true post-event kinematics (speed_A_post, speed_B_post, ke_ratio, dv_A, dv_B, v_cm_post); the model detects the violation beyond speed only if the paired group-bootstrap CI of model − kinematic AUC (selected cell) is above 0. Speed-matched: groups whose violator's KE is within the tolerance of its legal twin's. Norm routes: ridge to the true Δp (fitted_norm) or to (dv_A, dv_B) composed with true masses (composed_norm), score |Δp̂|; 'conserved' is a threshold on a norm, which a linear logistic probe cannot express.

| contrast | n groups | held-out AUC, overall (selected cell) [CI] | frame pool AUC (layer) | ROI pool AUC (layer) | ROI mask floor | kinematic AUC | ΔAUC vs kinematic | model − kinematic, selected cell [paired CI] | speed-matched: n groups, model / kinematic AUC, diff [CI] | ΔAUC vs mask floor | ROI − mask floor [paired CI] | fitted_norm |Δp̂| AUC (held-out) | composed_norm AUC (held-out) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mom_violate_e015 vs conserve_e015 | 64 | 0.566 (snr+8/L9/roi_pair.post) [0.571, 0.663] | 0.543 (snr+0/L24) | 0.586 (snr+8/L9) | 0.566 | 0.680 | -0.114 | -0.068 [-0.142, +0.016] | 7, 0.520 / 0.520, +0.000 [-0.255, +0.214] | +0.000 | +0.046 [-0.009, +0.110] | 0.492 (snr-8/L21/roi_pair.post) | 0.492 (snr+8/L29/frame.post) | not detected beyond the kinematic baseline |
| mom_violate_e1 vs conserve_e1 | 64 | 0.542 (snr+0/L12/roi_pair.post) [0.536, 0.644] | 0.542 (snr-8/L6) | 0.555 (snr+0/L12) | 0.559 | 0.634 | -0.092 | -0.044 [-0.121, +0.032] | 6, 0.438 / 0.472, -0.035 [-0.319, +0.222] | -0.018 | +0.031 [-0.039, +0.105] | 0.527 (snr+8/L12/frame.post) | 0.490 (snr+0/L18/frame.post) | not detected beyond the kinematic baseline |
| conserve_e1 vs conserve_e015 (legal-vs-legal reference) | 64 | 0.966 (snr+0/L12/roi_pair.post) [0.948, 0.990] | 0.853 (snr-8/L6) | 0.966 (snr+0/L12) | 0.602 | 0.996 | -0.031 | -0.025 [-0.048, -0.007] |  | +0.364 | +0.370 [+0.319, +0.406] | 0.564 (snr-8/L3/frame.post) | 0.514 (snr+0/L29/frame.post) | not detected beyond the kinematic baseline |

### Transfer (train on one contrast, score held-out groups of the other)

Compare with the kinematic transfer AUC: F1 violators are slower in both contrasts, so a speed shortcut can transfer too; only model transfer above the kinematic baseline is evidence.

| train -> test | model cell (selected on train) | model transfer AUC | kinematic transfer AUC | test within-contrast AUC | fitted_norm transfer AUC (within) | composed_norm transfer AUC (within) |
|---|---|---|---|---|---|---|
| mom_violate_e1 vs conserve_e1 -> mom_violate_e015 vs conserve_e015 | snr+0/L12/roi_pair.post | 0.550 ± 0.011 | 0.614 | 0.566 | 0.497 (0.492) | 0.502 (0.492) |
| mom_violate_e015 vs conserve_e015 -> mom_violate_e1 vs conserve_e1 | snr+8/L9/roi_pair.post | 0.559 ± 0.014 | 0.665 | 0.542 | 0.497 (0.527) | 0.490 (0.490) |
