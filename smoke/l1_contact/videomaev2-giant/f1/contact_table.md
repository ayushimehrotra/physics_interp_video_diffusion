### videomaev2-giant (n_groups 64, spaces 1)

| pool | option | space | probe R² floor | η floor | η blind | margin (blind−floor) | η forecast | S_mom [95% CI] | interaction rate fc/floor | mask-only R² (A, B) | S_mom informative |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frame | P-real-mm | final | 0.025 | 0.726 | 0.469 | -0.257 | 0.804 | — [—, —] | 1.00 / 1.00 | — (frame) | False |
| roi | P-real-mm | final | 0.681 | 0.462 | 0.469 | 0.007 | 0.737 | -1.00 [-1.00, -1.00] | 1.00 / 1.00 | 0.08, 0.03 | False |
| contact | P-real-mm | final | 0.210 | 0.573 | 0.469 | -0.104 | 0.791 | — [-1.00, -1.00] | 1.00 / 1.00 | -0.01, -0.04 | False |
| frame | P-pred-real | final | 0.258 | 0.579 | 0.469 | -0.110 | 0.630 | — [-1.00, -1.00] | 1.00 / 1.00 | — (frame) | False |
| roi | P-pred-real | final | 0.838 | 0.362 | 0.469 | 0.107 | 0.814 | -1.00 [-1.00, -1.00] | 1.00 / 1.00 | 0.08, 0.03 | True |
| contact | P-pred-real | final | 0.372 | 0.501 | 0.469 | -0.032 | 0.555 | — [-1.00, 0.18] | 1.00 / 1.00 | -0.01, -0.04 | False |
| frame | P-real-mm+cal | final | 0.018 | 0.787 | 0.469 | -0.318 | 0.824 | — [—, —] | 1.00 / 1.00 | — (frame) | False |
| roi | P-real-mm+cal | final | 0.636 | 0.534 | 0.469 | -0.065 | 0.739 | — [-1.00, -1.00] | 1.00 / 1.00 | 0.08, 0.03 | False |
| contact | P-real-mm+cal | final | 0.192 | 0.645 | 0.469 | -0.176 | 0.814 | — [—, —] | 1.00 / 1.00 | -0.01, -0.04 | False |
| frame | P-pred-real+cal | final | 0.238 | 0.624 | 0.469 | -0.155 | 0.570 | — [-1.00, -1.00] | 1.00 / 1.00 | — (frame) | False |
| roi | P-pred-real+cal | final | 0.826 | 0.366 | 0.469 | 0.103 | 0.817 | -1.00 [-1.00, -1.00] | 1.00 / 1.00 | 0.08, 0.03 | True |
| contact | P-pred-real+cal | final | 0.343 | 0.502 | 0.469 | -0.033 | 0.531 | — [-1.00, 1.00] | 1.00 / 1.00 | -0.01, -0.04 | False |
