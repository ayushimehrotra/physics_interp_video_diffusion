# L2B — videomaev2-giant on f1

n_groups = 64 (grouped 5-fold x 4 seeds cross-fitting (spec §5: allowed below 100 groups); ~13 test groups per fold); variants searched = 24 (layers x 2 pools); 500 within-group label-swap permutations, max over the grid; layer selected on half the groups, AUC on the other half (100 evaluations). L2b is NOT motion-adjusted: a probe can separate twins by how much the balls move, so read each AUC against the legal-vs-legal reference row (two legal outcomes, different motion) and the ROI mask floor; the encoded-but-unused index inherits that confound.

| contrast | n groups | held-out AUC, overall (selected cell) | frame pool AUC (layer) | ROI pool AUC (layer) | ROI mask floor | p_corr (Holm) | L2 adj. win (held-out) | encoded-but-unused |
|---|---|---|---|---|---|---|---|---|
| mom_violate_e1 vs conserve_e1 | 64 | 0.574 (L32/frame.post) | 0.588 (L32) | 0.546 (L24) | 0.504 | 0.012 (0.024) | 0.50 | +0.07 |
| mom_violate_e015 vs conserve_e015 | 64 | 0.578 (L12/roi_pair.post) | 0.522 (L16) | 0.578 (L12) | 0.475 | 0.018 (0.024) | 0.51 | +0.07 |
| energy_create vs conserve_e015 | 64 | 0.948 (L24/roi_pair.post) | 0.940 (L24) | 0.956 (L24) | 0.607 | 0.002 (0.00599) | 0.47 | +0.48 |
| conserve_e1 vs conserve_e015 (legal-vs-legal reference) | 64 | 0.901 (L24/roi_pair.post) | 0.897 (L24) | 0.904 (L24) | 0.583 | 0.002 (nan) | 0.53 | +0.38 |
