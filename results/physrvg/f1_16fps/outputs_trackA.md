# Output track A - physrvg on f1

Readouts: `/workspace/physlaw/cache/physrvg/f1`

Gate: {'n': 20, 'pass_rate': 0.45, 'min_pass_rate': 0.5, 'passed': False, 'reasons': {'lost_after_context': 3, 'bad_fit': 2, 'ok': 9, 'contact_at_edge': 6}}

| clips | n | measurable | drop reasons | median epsilon | median eta | median antiparallel | median normal_A | median normal_B | violations |
|---|---|---|---|---|---|---|---|---|---|
| generated | 20 | 9 | {'lost_after_context': 3, 'bad_fit': 2, 'contact_at_edge': 6} | -0.747 | 0.684 | 0.636 | 0.751 | 0.968 |  |
| real conserve_e1 | 20 | 16 | {'contact_at_edge': 4} | 0.759 | 0.068 | 1.000 | 1.000 | 1.000 |  |
| real conserve_e015 | 20 | 14 | {'contact_at_edge': 3, 'lost_after_context': 3} | -0.001 | 0.056 | 0.997 | 1.000 | 1.000 |  |

### Energy: share with epsilon <= 1 + 0.1

| clips | n | share eps <= 1 + tol [95% CI] | median eps | true eps |
|---|---|---|---|---|
| generated | 9 | 1.00 [+1.00, +1.00] | -0.747 | - |
| real legal (pooled) | 30 | 1.00 | 0.220 | - |
| real creating (pooled) | 0 | - | - | - |
| real conserve_e015 | 14 | 1.00 | -0.001 | 0.150 |
| real conserve_e1 | 16 | 1.00 | 0.759 | 1.000 |

### Momentum (S_mom package, kappa 0.25, mass ratio >= 1.25)

Interaction rate = measurable generated clips whose momentum exchange D >= kappa x the weakest legal collision (real clips in brackets); eta = |sum m dv| / sum |m dv| pooled over gated clips (generated / real readout floor / mass-blind rule); S_mom = 1 - (eta_gen - eta_real) / (eta_blind - eta_real).

| legal twin | interaction rate (real) | scored clips / groups | eta gen / real / blind | eta real / blind, all real clips | S_mom [95% CI] |
|---|---|---|---|---|---|
| conserve_e1 | 1.00 (1.00) | 7 / 7 | 0.687 / 0.078 / 0.573 | 0.047 / 0.508 (14) | -0.23 [-0.99, +0.02] |
| conserve_e015 | 1.00 (1.00) | 7 / 7 | 0.687 / 0.152 / 0.573 | 0.115 / 0.436 (12) | -0.27 [-1.00, +0.03] |
| legal | 1.00 (1.00) | 14 / 7 | 0.687 / 0.106 / 0.573 | 0.071 / 0.485 (26) | -0.25 [-1.00, +0.02] |
