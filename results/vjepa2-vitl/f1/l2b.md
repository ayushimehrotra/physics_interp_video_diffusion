# L2B — vjepa2-vitl on f1

n_groups = 64 (grouped 5-fold x 4 seeds cross-fitting (spec §5: allowed below 100 groups); ~13 test groups per fold); variants searched = 26 (layers x 2 pools); 500 within-group label-swap permutations, max over the grid; layer selected on half the groups, AUC on the other half (100 evaluations). L2b is NOT motion-adjusted: a probe can separate twins by how much the balls move, so read each AUC against the legal-vs-legal reference row (two legal outcomes, different motion) and the ROI mask floor; the encoded-but-unused index inherits that confound.

| contrast | n groups | held-out AUC, overall (selected cell) | frame pool AUC (layer) | ROI pool AUC (layer) | ROI mask floor | p_corr (Holm) | L2 adj. win (held-out) | encoded-but-unused |
|---|---|---|---|---|---|---|---|---|
| mom_violate_e1 vs conserve_e1 | 64 | 0.574 (L12/roi_pair.post) | 0.562 (L12) | 0.594 (L12) | 0.568 | 0.0299 (0.0299) | 0.66 | -0.08 |
| mom_violate_e015 vs conserve_e015 | 64 | 0.584 (L4/roi_pair.post) | 0.548 (L16) | 0.592 (L4) | 0.455 | 0.012 (0.024) | 0.70 | -0.12 |
| energy_create vs conserve_e015 | 64 | 0.971 (L14/roi_pair.post) | 0.969 (L16) | 0.976 (L14) | 0.645 | 0.002 (0.00599) | 0.59 | +0.38 |
| conserve_e1 vs conserve_e015 (legal-vs-legal reference) | 64 | 0.940 (L16/roi_pair.post) | 0.939 (L16) | 0.949 (L16) | 0.564 | 0.002 (nan) | 0.57 | +0.37 |
