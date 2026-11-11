### vjepa2-vitl (n_groups 64, spaces 1)

| pool | option | space | probe R² floor | η floor | η blind | margin (blind−floor) | η forecast | S_mom [95% CI] | interaction rate fc/floor | mask-only R² (A, B) | S_mom informative |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frame | P-real-mm | final | 0.481 | 0.467 | 0.469 | 0.001 | 0.714 | -1.00 [-1.00, -1.00] | 1.00 / 1.00 | — (frame) | False |
| roi | P-real-mm | final | 0.895 | 0.297 | 0.469 | 0.171 | 0.706 | -1.00 [-1.00, -0.68] | 1.00 / 1.00 | 0.24, 0.12 | True |
| contact | P-real-mm | final | 0.489 | 0.407 | 0.469 | 0.062 | 0.716 | -1.00 [-1.00, -1.00] | 1.00 / 1.00 | -0.03, -0.06 | False |
| frame | P-pred-real | final | 0.456 | 0.494 | 0.469 | -0.025 | 0.704 | — [-1.00, -1.00] | 1.00 / 1.00 | — (frame) | False |
| roi | P-pred-real | final | 0.871 | 0.334 | 0.469 | 0.135 | 0.684 | -1.00 [-1.00, -0.79] | 1.00 / 1.00 | 0.24, 0.12 | True |
| contact | P-pred-real | final | 0.465 | 0.447 | 0.469 | 0.022 | 0.661 | -1.00 [-1.00, -1.00] | 1.00 / 1.00 | -0.03, -0.06 | False |
| frame | P-real-mm+cal | final | 0.458 | 0.481 | 0.469 | -0.012 | 0.721 | — [-1.00, -1.00] | 1.00 / 1.00 | — (frame) | False |
| roi | P-real-mm+cal | final | 0.887 | 0.306 | 0.469 | 0.163 | 0.700 | -1.00 [-1.00, -0.72] | 1.00 / 1.00 | 0.24, 0.12 | True |
| contact | P-real-mm+cal | final | 0.468 | 0.421 | 0.469 | 0.048 | 0.726 | -1.00 [-1.00, -1.00] | 1.00 / 1.00 | -0.03, -0.06 | False |
| frame | P-pred-real+cal | final | 0.433 | 0.512 | 0.469 | -0.043 | 0.719 | — [-1.00, -1.00] | 1.00 / 1.00 | — (frame) | False |
| roi | P-pred-real+cal | final | 0.861 | 0.357 | 0.469 | 0.112 | 0.682 | -1.00 [-1.00, -0.87] | 1.00 / 1.00 | 0.24, 0.12 | True |
| contact | P-pred-real+cal | final | 0.441 | 0.486 | 0.469 | -0.017 | 0.689 | — [-1.00, -1.00] | 1.00 / 1.00 | -0.03, -0.06 | False |
