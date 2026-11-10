# L2B — cogvideox-2b on f1

n_groups = 64 (grouped 5-fold x 4 seeds cross-fitting (spec §5: allowed below 100 groups); ~13 test groups per fold); variants searched = 88 (layers x 2 pools); 500 within-group label-swap permutations, max over the grid; layer selected on half the groups, AUC on the other half (100 evaluations). L2b is NOT motion-adjusted: a probe can separate twins by how much the balls move, so read each AUC against the legal-vs-legal reference row (two legal outcomes, different motion) and the ROI mask floor; the encoded-but-unused index inherits that confound.

| contrast | n groups | held-out AUC, overall (selected cell) | frame pool AUC (layer) | ROI pool AUC (layer) | ROI mask floor | p_corr (Holm) | L2 adj. win (held-out) | encoded-but-unused |
|---|---|---|---|---|---|---|---|---|
| mom_violate_e1 vs conserve_e1 | 64 | 0.552 (snr+8/L27/frame.post) | 0.555 (snr+8/L27) | 0.517 (snr-20/L18) | 0.484 | 0.116 (0.116) | 0.54 | +0.01 |
| mom_violate_e015 vs conserve_e015 | 64 | 0.573 (snr+8/L18/frame.post) | 0.574 (snr+8/L18) | 0.536 (snr-20/L12) | 0.513 | 0.00599 (0.012) | 0.63 | -0.06 |
| energy_create vs conserve_e015 | 64 | 0.935 (snr-8/L15/roi_pair.post) | 0.927 (snr+0/L6) | 0.939 (snr-8/L15) | 0.661 | 0.002 (0.00599) | 0.55 | +0.39 |
| conserve_e1 vs conserve_e015 (legal-vs-legal reference) | 64 | 0.885 (snr-8/L12/roi_pair.post) | 0.816 (snr+0/L6) | 0.887 (snr-8/L12) | 0.599 | 0.002 (nan) | 0.57 | +0.31 |
