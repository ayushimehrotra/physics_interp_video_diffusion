# Output-level evaluation — cogvideox-2b on f1

Gate: {'n': 20, 'pass_rate': 0.0, 'min_pass_rate': 0.3, 'passed': False, 'reasons': {'lost_after_context': 6, 'contact_at_edge': 14}}

| legal twin | generated measurable | drop reasons | real measurable | r_p gen / floor | r_E gen / floor | E_out gen / floor | S_outcome [95% CI] |
|---|---|---|---|---|---|---|---|
| conserve_e1 | 1/64 | {'lost_after_context': 19, 'contact_at_edge': 42, 'bad_fit': 2} | 60/64 | 1.261 / 0.044 | 2.832 / 0.005 | 1.577 / 0.048 | -0.61 [-0.59, -0.59] |
| conserve_e015 | 1/64 | {'lost_after_context': 19, 'contact_at_edge': 42, 'bad_fit': 2} | 55/64 | 1.261 / 0.054 | 2.832 / 0.000 | 3.221 / 0.069 | -1.00 [-1.00, -1.00] |
