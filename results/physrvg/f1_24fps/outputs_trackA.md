# Output track A - physrvg on f1

Readouts: `cache/physrvg/f1_24fps_trackA`

Gate: {'n': 20, 'pass_rate': 0.5, 'min_pass_rate': 0.5, 'passed': True, 'reasons': {'lost_after_context': 4, 'contact_at_edge': 5, 'ok': 10, 'extra_objects': 1}}

| clips | n | measurable | drop reasons | median epsilon | median eta | median antiparallel | median normal_A | median normal_B | violations |
|---|---|---|---|---|---|---|---|---|---|
| generated | 256 | 174 | {'lost_after_context': 26, 'contact_at_edge': 46, 'extra_objects': 7, 'bad_fit': 3} | -0.591 | 0.709 | 0.627 | 0.835 | 0.985 |  |
| real conserve_e1 | 64 | 48 | {'contact_at_edge': 15, 'no_collision': 1} | 0.310 | 0.101 | 0.997 | 0.999 | 1.000 |  |
| real conserve_e015 | 64 | 48 | {'lost_after_context': 7, 'contact_at_edge': 8, 'no_collision': 1} | -0.171 | 0.078 | 0.999 | 1.000 | 1.000 |  |

### Energy: share with epsilon <= 1 + 0.1

| clips | n | share eps <= 1 + tol [95% CI] | median eps | true eps |
|---|---|---|---|---|
| generated | 174 | 1.00 [+1.00, +1.00] | -0.591 | - |
| real legal (pooled) | 96 | 1.00 | 0.145 | - |
| real creating (pooled) | 0 | - | - | - |
| real conserve_e015 | 48 | 1.00 | -0.171 | 0.150 |
| real conserve_e1 | 48 | 1.00 | 0.310 | 1.000 |

### Momentum (S_mom package, kappa 0.25, mass ratio >= 1.25)

Interaction rate = measurable generated clips whose momentum exchange D >= kappa x the weakest legal collision (real clips in brackets); eta = |sum m dv| / sum |m dv| pooled over gated clips (generated / real readout floor / mass-blind rule); S_mom = 1 - (eta_gen - eta_real) / (eta_blind - eta_real).

| legal twin | interaction rate (real) | scored clips / groups | eta gen / real / blind | eta real / blind, all real clips | S_mom [95% CI] |
|---|---|---|---|---|---|
| conserve_e1 | 0.98 (1.00) | 138 / 39 | 0.704 / 0.114 / 0.466 | 0.111 / 0.421 (40) | -0.68 [-1.00, -0.36] |
| conserve_e015 | 0.98 (1.00) | 138 / 39 | 0.704 / 0.107 / 0.466 | 0.094 / 0.421 (40) | -0.66 [-1.00, -0.34] |
| legal | 0.98 (1.00) | 276 / 39 | 0.704 / 0.111 / 0.466 | 0.104 / 0.421 (80) | -0.67 [-1.00, -0.36] |
