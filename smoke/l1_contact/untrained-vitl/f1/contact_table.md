### untrained-vitl (n_groups 64, spaces 1)

| pool | option | space | probe R² floor | η floor | η blind | margin (blind−floor) | η forecast | S_mom [95% CI] | interaction rate fc/floor | mask-only R² (A, B) | S_mom informative |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frame | P-real-mm | final | -0.018 | 0.807 | 0.469 | -0.338 | 0.821 | — [—, —] | 1.00 / 1.00 | — (frame) | False |
| roi | P-real-mm | final | 0.581 | 0.516 | 0.469 | -0.047 | 0.825 | — [-1.00, -1.00] | 1.00 / 1.00 | 0.24, 0.12 | False |
| contact | P-real-mm | final | -0.014 | 0.809 | 0.469 | -0.340 | 0.823 | — [—, —] | 1.00 / 1.00 | -0.03, -0.06 | False |
| frame | P-real-mm+cal | final | -0.024 | 0.808 | 0.469 | -0.339 | 0.823 | — [—, —] | 1.00 / 1.00 | — (frame) | False |
| roi | P-real-mm+cal | final | 0.542 | 0.623 | 0.469 | -0.154 | 0.799 | — [—, —] | 1.00 / 1.00 | 0.24, 0.12 | False |
| contact | P-real-mm+cal | final | -0.005 | 0.810 | 0.469 | -0.341 | 0.825 | — [—, —] | 1.00 / 1.00 | -0.03, -0.06 | False |
