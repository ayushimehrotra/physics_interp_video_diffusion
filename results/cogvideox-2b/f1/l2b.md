# L2B — cogvideox-2b on f1

n_groups = 64 (grouped 5-fold x 4 seeds cross-fitting (spec §5: allowed below 100 groups); ~13 test groups per fold); variants searched = 88 (layers x 2 pools); layer selected on half the groups, AUC on the other half (100 evaluations); CI: group bootstrap of the selected cell's out-of-fold AUC. a probe can separate twins by how much the objects move, so read each AUC against the legal-vs-legal reference row (two legal outcomes, different motion), the ROI mask floor, the kinematic baseline and the speed-matched stratum. Energy is not scored here (spec §5.0.3 R5; see L2-rep).

Kinematic baseline: the same probe protocol on the true post-event kinematics (speed_A_post, speed_B_post, ke_ratio, dv_A, dv_B, v_cm_post); the model detects the violation beyond speed only if the paired group-bootstrap CI of model − kinematic AUC (selected cell) is above 0. Speed-matched: groups whose violator's KE is within the tolerance of its legal twin's. Norm routes: ridge to the true Δp (fitted_norm) or to (dv_A, dv_B) composed with true masses (composed_norm), score |Δp̂|; 'conserved' is a threshold on a norm, which a linear logistic probe cannot express.

| contrast | n groups | held-out AUC, overall (selected cell) [CI] | frame pool AUC (layer) | ROI pool AUC (layer) | ROI mask floor | kinematic AUC | ΔAUC vs kinematic | model − kinematic, selected cell [paired CI] | speed-matched: n groups, model / kinematic AUC, diff [CI] | ΔAUC vs mask floor | ROI − mask floor [paired CI] | fitted_norm |Δp̂| AUC (held-out) | composed_norm AUC (held-out) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mom_violate_e015 vs conserve_e015 | 64 | 0.574 (snr+0/L9/frame.post) [0.559, 0.654] | 0.577 (snr+8/L29) | 0.540 (snr-20/L12) | 0.513 | 0.680 | -0.106 | -0.074 [-0.138, -0.004] | 7, 0.490 / 0.520, -0.031 [-0.225, +0.051] | +0.061 | +0.059 [+0.007, +0.114] | 0.503 (snr+0/L15/roi_pair.post) | 0.490 (snr-20/L21/frame.post) | not detected beyond the kinematic baseline |
| mom_violate_e1 vs conserve_e1 | 64 | 0.546 (snr-20/L0/frame.post) [0.532, 0.643] | 0.550 (snr-20/L0) | 0.518 (snr-20/L18) | 0.484 | 0.634 | -0.088 | -0.048 [-0.132, +0.039] | 6, 0.500 / 0.472, +0.028 [-0.264, +0.278] | +0.062 | +0.076 [-0.020, +0.174] | 0.528 (snr-20/L29/roi_pair.post) | 0.495 (snr-20/L29/frame.post) | not detected beyond the kinematic baseline |
| conserve_e1 vs conserve_e015 (legal-vs-legal reference) | 64 | 0.885 (snr-8/L12/roi_pair.post) [0.860, 0.946] | 0.817 (snr+0/L6) | 0.886 (snr-8/L12) | 0.599 | 0.996 | -0.112 | -0.092 [-0.136, -0.050] |  | +0.285 | +0.305 [+0.242, +0.355] | 0.567 (snr-20/L9/frame.post) | 0.519 (snr-20/L24/frame.post) | not detected beyond the kinematic baseline |

### Transfer (train on one contrast, score held-out groups of the other)

Compare with the kinematic transfer AUC: F1 violators are slower in both contrasts, so a speed shortcut can transfer too; only model transfer above the kinematic baseline is evidence.

| train -> test | model cell (selected on train) | model transfer AUC | kinematic transfer AUC | test within-contrast AUC | fitted_norm transfer AUC (within) | composed_norm transfer AUC (within) |
|---|---|---|---|---|---|---|
| mom_violate_e1 vs conserve_e1 -> mom_violate_e015 vs conserve_e015 | snr-20/L0/frame.post | 0.528 ± 0.011 | 0.614 | 0.574 | 0.509 (0.503) | 0.489 (0.490) |
| mom_violate_e015 vs conserve_e015 -> mom_violate_e1 vs conserve_e1 | snr+0/L9/frame.post | 0.555 ± 0.017 | 0.665 | 0.546 | 0.480 (0.528) | 0.492 (0.495) |
