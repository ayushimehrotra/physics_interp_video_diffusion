# Output track B - wan2.2-i2v-a14b on f1

Readouts: `/tmp/claude-0/-root/1e571780-01d0-4d88-a090-f041787bcada/scratchpad/levels/wan_trackB`; laws: momentum, energy, com, angular_momentum

## Gate: strict

| clips | n | measurable | drop reasons | median e | median r_momentum | median ke_ratio | median res_momentum | median res_energy | median res_com | median res_angular_momentum | violations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| generated | 22 | 0 | {'extra_objects': 2, 'short_window': 5, 'size_change': 2, 'no_collision': 1, 'fit_residual,impulse_not_opposite,still_approaching,curved_path': 5, 'fit_residual,impulse_not_opposite,curved_path': 3, 'fit_residual,curved_path': 2, 'fit_residual,still_approaching,curved_path': 2} | - | - | - | - | - | - | - |  |
| real conserve_e015 | 64 | 61 | {'still_approaching': 2, 'impulse_not_opposite,still_approaching': 1} | 0.141 | 0.041 | 0.752 | 0.041 | 0.000 | 0.035 | 0.024 |  |
| real conserve_e1 | 64 | 64 | {} | 1.001 | 0.031 | 0.999 | 0.031 | 0.000 | 0.023 | 0.021 |  |
| real energy_create | 64 | 64 | {} | 1.500 | 0.029 | 1.319 | 0.029 | 0.319 | 0.024 | 0.027 |  |
| real mom_violate_e015 | 64 | 59 | {'never_separate': 1, 'still_approaching': 3, 'short_window': 1} | 0.141 | 0.330 | 0.613 | 0.330 | 0.000 | 0.290 | 0.144 |  |
| real mom_violate_e1 | 64 | 64 | {} | 1.000 | 0.610 | 0.839 | 0.610 | 0.000 | 0.491 | 0.270 |  |

Real-clip validation (§6.1): 312/320 measurable.

Measurable generated clips: none

## Gate: loose

| clips | n | measurable | drop reasons | median e | median r_momentum | median ke_ratio | median res_momentum | median res_energy | median res_com | median res_angular_momentum | violations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| generated | 22 | 8 | {'extra_objects': 2, 'short_window': 5, 'size_change': 2, 'no_collision': 1, 'jitter': 4} | -0.152 | 0.998 | 0.315 | 0.998 | 0.000 | 0.723 | 0.320 | {'impulse_not_opposite': 7, 'still_approaching': 4, 'curved_path': 8, 'self_acceleration': 5} |
| real conserve_e015 | 64 | 64 | {} | 0.139 | 0.042 | 0.744 | 0.042 | 0.000 | 0.037 | 0.024 | {'impulse_not_opposite': 1} |
| real conserve_e1 | 64 | 64 | {} | 1.001 | 0.031 | 0.999 | 0.031 | 0.000 | 0.023 | 0.021 |  |
| real energy_create | 64 | 64 | {} | 1.500 | 0.029 | 1.319 | 0.029 | 0.319 | 0.024 | 0.027 |  |
| real mom_violate_e015 | 64 | 62 | {'never_separate': 1, 'short_window': 1} | 0.140 | 0.329 | 0.613 | 0.329 | 0.000 | 0.288 | 0.144 |  |
| real mom_violate_e1 | 64 | 64 | {} | 1.000 | 0.610 | 0.839 | 0.610 | 0.000 | 0.491 | 0.270 |  |

Real-clip validation (§6.1): 318/320 measurable.

Measurable generated clips: f1_100002/seed0, f1_100002/seed1, f1_100003/seed0, f1_100003/seed1, f1_100005/seed1, f1_100007/seed0, f1_100009/seed0, f1_100011/seed0

