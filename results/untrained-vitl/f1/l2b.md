# L2B — untrained-vitl on f1

n_groups = 64 (grouped 5-fold x 4 seeds cross-fitting (spec §5: allowed below 100 groups); ~13 test groups per fold); variants searched = 26 (layers x 2 pools); 500 within-group label-swap permutations, max over the grid; layer selected on half the groups, AUC on the other half (100 evaluations). L2b is NOT motion-adjusted: a probe can separate twins by how much the balls move, so read each AUC against the legal-vs-legal reference row (two legal outcomes, different motion) and the ROI mask floor; the encoded-but-unused index inherits that confound.

| contrast | n groups | held-out AUC, overall (selected cell) | frame pool AUC (layer) | ROI pool AUC (layer) | ROI mask floor | p_corr (Holm) | L2 adj. win (held-out) | encoded-but-unused |
|---|---|---|---|---|---|---|---|---|
| mom_violate_e1 vs conserve_e1 | 64 | 0.547 (L12/roi_pair.post) | 0.479 (L8) | 0.547 (L12) | 0.568 | 0.154 (0.154) | 0.43 | +0.11 |
| mom_violate_e015 vs conserve_e015 | 64 | 0.627 (L12/roi_pair.post) | 0.515 (L20) | 0.627 (L12) | 0.455 | 0.0279 (0.0559) | 0.45 | +0.18 |
| energy_create vs conserve_e015 | 64 | 0.744 (L0/frame.post) | 0.749 (L0) | 0.762 (L6) | 0.645 | 0.002 (0.00599) | 0.51 | +0.24 |
| conserve_e1 vs conserve_e015 (legal-vs-legal reference) | 64 | 0.727 (L2/roi_pair.post) | 0.711 (L0) | 0.742 (L2) | 0.564 | 0.002 (nan) | 0.46 | +0.27 |
