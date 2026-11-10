# Leaderboard (spec §8)

Blank = level not run. Cells list sub-scores as key value; see physlaw/analysis/aggregate.py for sources.

## f1

| model | l0 | l1 | l2 | l2b | encoded_unused | l3 | l4 | outputs_trackA | outputs_trackB |
|---|---|---|---|---|---|---|---|---|---|
| cogvideox-2b | mean_r2 +0.97<br>n_targets 20 | S_outcome.conserve_e1 -0.08<br>S_momentum.conserve_e1 +0.04<br>S_outcome.conserve_e015 -1.00<br>S_momentum.conserve_e015 +0.04<br>variant median over 12 spaces (frame_mm) | mom_violate_e1 vs conserve_e1 +0.45<br>mom_violate_e1 vs conserve_e1 [energy_physical] +0.37<br>mom_violate_e015 vs conserve_e015 +0.55<br>energy_create vs conserve_e015 +0.61<br>conserve_e1 vs conserve_e015 +0.52 |  |  |  |  | measurable.conserve_e1 1/64<br>S_outcome.conserve_e1 -0.61<br>measurable.conserve_e015 1/64<br>S_outcome.conserve_e015 -1.00<br>gate_passed False |  |
| untrained-vitl | mean_r2 +0.33<br>n_targets 20 |  | mom_violate_e1 vs conserve_e1 +0.50<br>mom_violate_e1 vs conserve_e1 [energy_physical] +0.51<br>mom_violate_e015 vs conserve_e015 +0.52<br>energy_create vs conserve_e015 +0.55<br>conserve_e1 vs conserve_e015 +0.50 |  |  |  |  |  |  |
| videomaev2-giant | mean_r2 +0.89<br>n_targets 20 | S_outcome.conserve_e1 -0.66<br>S_momentum.conserve_e1 +0.18<br>S_momentum.conserve_e015 +0.19<br>variant frame_mm | mom_violate_e1 vs conserve_e1 +0.42<br>mom_violate_e1 vs conserve_e1 [energy_physical] +0.47<br>mom_violate_e015 vs conserve_e015 +0.50<br>energy_create vs conserve_e015 +0.56<br>conserve_e1 vs conserve_e015 +0.44 |  |  |  |  |  |  |
| vjepa2-vitl | mean_r2 +0.91<br>n_targets 20 | S_outcome.conserve_e1 +0.26<br>S_momentum.conserve_e1 +0.43<br>S_outcome.conserve_e015 -1.00<br>S_momentum.conserve_e015 +0.40<br>variant frame_mm | mom_violate_e1 vs conserve_e1 +0.72<br>mom_violate_e1 vs conserve_e1 [energy_physical] +0.77<br>mom_violate_e015 vs conserve_e015 +0.73<br>energy_create vs conserve_e015 +0.59<br>conserve_e1 vs conserve_e015 +0.47 |  |  |  |  |  |  |
| wan2.2-i2v-a14b |  |  |  |  |  |  |  |  | measurable.strict 0/22<br>measurable.loose 8/22<br>res_momentum.loose +1.00<br>res_energy.loose +0.00<br>res_com.loose +0.72<br>res_angular_momentum.loose +0.32<br>no_violation.loose 0/22 |
