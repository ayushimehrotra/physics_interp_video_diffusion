### cogvideox-2b (n_groups 64, spaces 12)

| pool | option | space | probe R² floor | η floor | η blind | margin (blind−floor) | η forecast | S_mom [95% CI] | interaction rate fc/floor | mask-only R² (A, B) | S_mom informative |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frame | P-real-mm | snr+0/L15 | 0.369 | 0.444 | 0.469 | 0.025 | 0.806 | -1.00 [-1.00, -1.00] | 1.00 / 1.00 | — (frame) | False |
| roi | P-real-mm | snr-20/L6 | 0.866 | 0.301 | 0.469 | 0.168 | 0.566 | -0.58 [-1.00, -0.13] | 1.00 / 1.00 | 0.42, 0.39 | True |
| contact | P-real-mm | snr+0/L15 | 0.390 | 0.499 | 0.469 | -0.030 | 0.812 | — [-1.00, -1.00] | 1.00 / 1.00 | -0.02, -0.05 | False |
| frame | P-real-mm+cal | snr+0/L15 | 0.362 | 0.487 | 0.469 | -0.018 | 0.811 | — [-1.00, -1.00] | 1.00 / 1.00 | — (frame) | False |
| roi | P-real-mm+cal | snr-20/L6 | 0.859 | 0.301 | 0.469 | 0.168 | 0.566 | -0.58 [-1.00, -0.13] | 1.00 / 1.00 | 0.42, 0.39 | True |
| contact | P-real-mm+cal | snr+0/L15 | 0.378 | 0.507 | 0.468 | -0.038 | 0.808 | — [-1.00, -1.00] | 1.00 / 0.99 | -0.02, -0.05 | False |
