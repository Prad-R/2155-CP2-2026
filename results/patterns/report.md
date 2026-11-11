# Pattern analysis

## A. Structure of the score values

| advisor | distinct values | values are k/N for N= | share = 0 | share = 1 | min gap | median gap | share >= 0.75 |
|---|---|---|---|---|---|---|---|
| Wellness | 488 | None | 0.001 | 0.000 | 1.72e-05 | 1.00e-03 | 0.250 |
| Tax | 360 | None | 0.000 | 0.000 | 7.37e-18 | 1.50e-03 | 0.238 |
| Transportation | 241 | None | 0.000 | 0.000 | 2.70e-06 | 2.00e-03 | 0.249 |
| Business | 818 | None | 0.000 | 0.000 | 4.51e-17 | 1.00e-03 | 0.246 |

## B. District counts: score by count, and the range valid grids occupy

| advisor | district | Spearman(count, score) | count range of valid grids (10-90%) | count with highest valid rate |
|---|---|---|---|---|
| Wellness | Res | -0.01 | 5-11 (median 9) | 9 (31% valid) |
| Wellness | Ind | -0.19 | 4-14 (median 9) | 1 (50% valid) |
| Wellness | Com | +0.06 | 7-16 (median 11) | 16 (33% valid) |
| Wellness | Park | +0.65 | 8-13 (median 10) | 14 (77% valid) |
| Wellness | Off | -0.11 | 4-16 (median 10) | 29 (33% valid) |
| Tax | Res | +0.27 | 4-13 (median 10) | 16 (89% valid) |
| Tax | Ind | +0.01 | 4-19 (median 10) | 21 (48% valid) |
| Tax | Com | +0.26 | 8-21 (median 13) | 25 (94% valid) |
| Tax | Park | -0.70 | 0-8 (median 5) | 0 (100% valid) |
| Tax | Off | +0.00 | 3-20 (median 10) | 25 (43% valid) |
| Transportation | Res | +0.35 | 7-12 (median 10) | 11 (37% valid) |
| Transportation | Ind | -0.20 | 5-12 (median 9) | 9 (38% valid) |
| Transportation | Com | -0.07 | 6-14 (median 10) | 11 (32% valid) |
| Transportation | Park | +0.27 | 6-12 (median 9) | 10 (36% valid) |
| Transportation | Off | +0.02 | 6-17 (median 11) | 11 (34% valid) |
| Business | Res | -0.07 | 5-11 (median 9) | 10 (37% valid) |
| Business | Ind | +0.29 | 5-22 (median 12) | 27 (83% valid) |
| Business | Com | -0.56 | 4-11 (median 8) | 2 (66% valid) |
| Business | Park | -0.13 | 3-10 (median 8) | 0 (36% valid) |
| Business | Off | +0.27 | 4-22 (median 12) | 23 (57% valid) |

## C. Strongest single features per advisor (Spearman with the score)

**Wellness**

| feature | Spearman |
|---|---|
| n Park within 2 of a Com | +0.658 |
| n Park | +0.653 |
| n Park within 2 of a Park | +0.633 |
| n Park touching a Park | +0.611 |
| edge pairs Park-Park | +0.603 |
| mean dist Res->nearest Park | -0.568 |
| n Park within 2 of a Res | +0.541 |
| largest cluster Park | +0.537 |
| diagonal pairs Park-Park | +0.494 |
| diagonal pairs Com-Park | +0.492 |

**Tax**

| feature | Spearman |
|---|---|
| n Park | -0.695 |
| n Park within 2 of a Park | -0.657 |
| n Park touching a Park | -0.580 |
| edge pairs Park-Park | -0.563 |
| n Park within 2 of a Ind | -0.562 |
| n Park in ring 0 (0=edge) | -0.555 |
| n Park within 2 of a Com | -0.534 |
| n Park within 2 of a Off | -0.520 |
| diagonal pairs Park-Park | -0.508 |
| largest cluster Park | -0.489 |

**Transportation**

| feature | Spearman |
|---|---|
| n Res within 2 of a Com | +0.404 |
| n clusters Res | +0.403 |
| n Park touching a Res | +0.401 |
| n Res touching a Park | +0.397 |
| n Res within 2 of a Park | +0.397 |
| edge pairs Res-Park | +0.394 |
| n Park within 2 of a Res | +0.393 |
| n Res within 2 of a Off | +0.382 |
| n Res within 2 of a Ind | +0.353 |
| max dist Ind->nearest Res | -0.350 |

**Business**

| feature | Spearman |
|---|---|
| n Com | -0.557 |
| n Com within 2 of a Com | -0.550 |
| n Com touching a Com | -0.506 |
| edge pairs Com-Com | -0.496 |
| diagonal pairs Com-Com | -0.495 |
| n Com within 2 of a Park | -0.439 |
| n Com in ring 0 (0=edge) | -0.433 |
| largest cluster Com | -0.430 |
| n Com in ring 1 (0=edge) | -0.400 |
| mean dist Com->nearest Com | +0.400 |

## D. How much each kind of information explains (3-fold R2)

| advisor | counts (linear) | counts | counts+edge pairs | counts+distances | counts+clusters | counts+rings | all position-free | everything |
|---|---|---|---|---|---|---|---|---|
| Wellness | 0.485 | 0.544 | 0.744 | 0.864 | 0.640 | 0.597 | 0.887 | 0.892 |
| Tax | 0.760 | 0.881 | 0.969 | 0.952 | 0.902 | 0.893 | 0.963 | 0.963 |
| Transportation | 0.153 | 0.382 | 0.626 | 0.921 | 0.442 | 0.402 | 0.932 | 0.929 |
| Business | 0.503 | 0.944 | 0.968 | 0.954 | 0.965 | 0.940 | 0.962 | 0.961 |

## E. Where the blended model is wrong (out-of-fold residual = truth - prediction)

| advisor | sd, pred 0-0.25 | sd, pred 0.25-0.5 | sd, pred 0.5-0.65 | sd, pred 0.65-0.75 | sd, pred 0.75-0.85 | sd, pred 0.85-1 | pred>=0.75 but invalid | valid but pred<0.75 | features most correlated with residual (pred>=0.6) |
|---|---|---|---|---|---|---|---|---|---|
| Wellness | 0.052 | 0.102 | 0.114 | 0.096 | 0.074 | 0.045 | 8.3% | 22.5% | n Park within 2 of a Com (-0.21); n Com within 2 of a Park (-0.20); diagonal pairs Com-Park (-0.19) |
| Tax | 0.031 | 0.058 | 0.052 | 0.040 | 0.032 | 0.016 | 3.7% | 7.2% | n Park within 2 of a Park (+0.17); n Park (+0.17); max dist Park->nearest Res (+0.16) |
| Transportation | 0.040 | 0.061 | 0.066 | 0.059 | 0.047 | 0.031 | 6.4% | 11.4% | n clusters Com (-0.17); n Res within 2 of a Ind (-0.15); n clusters Ind (-0.15) |
| Business | 0.021 | 0.033 | 0.037 | 0.033 | 0.027 | 0.018 | 2.5% | 5.8% | n Res (+0.19); n Com within 2 of a Com (+0.18); n Com (+0.17) |

## F. What our optimised sets look like

| set | score (10-NN mean) | nearest | 10th nearest | mean over all pairs | limit from per-cell mix | limit from overall mix | cells per grid Res/Ind/Com/Park/Off | tightest advisor W/T/Tr/B (grids) | mean prediction W/T/Tr/B |
|---|---|---|---|---|---|---|---|---|---|
| b1 | 0.7540 | 0.745 | 0.756 | 0.804 | 0.796 | 0.798 | 10.1 / 10.2 / 10.0 / 8.0 / 10.7 | 74 / 13 / 3 / 10 | 0.905 / 0.834 / 0.937 / 0.842 |
| b2 | 0.7542 | 0.747 | 0.755 | 0.804 | 0.796 | 0.798 | 10.1 / 10.3 / 10.0 / 8.0 / 10.7 | 73 / 14 / 2 / 11 | 0.909 / 0.830 / 0.938 / 0.838 |
| b3 | 0.7531 | 0.743 | 0.755 | 0.804 | 0.796 | 0.798 | 10.1 / 10.2 / 10.0 / 8.0 / 10.7 | 75 / 13 / 3 / 9 | 0.905 / 0.833 / 0.929 / 0.842 |
| b4 | 0.7550 | 0.752 | 0.756 | 0.804 | 0.796 | 0.798 | 10.1 / 10.4 / 9.9 / 8.0 / 10.6 | 68 / 22 / 2 / 8 | 0.888 / 0.820 / 0.937 / 0.834 |
| x1 | 0.7534 | 0.743 | 0.756 | 0.804 | 0.796 | 0.798 | 10.1 / 10.2 / 10.0 / 7.9 / 10.7 | 66 / 18 / 4 / 12 | 0.911 / 0.830 / 0.937 / 0.840 |
| x2 | 0.7529 | 0.740 | 0.756 | 0.804 | 0.796 | 0.798 | 10.2 / 10.2 / 9.9 / 8.0 / 10.7 | 67 / 18 / 4 / 11 | 0.911 / 0.828 / 0.937 / 0.837 |
| x3 | 0.7529 | 0.740 | 0.755 | 0.804 | 0.796 | 0.798 | 10.2 / 10.4 / 10.0 / 7.9 / 10.5 | 61 / 22 / 4 / 13 | 0.910 / 0.830 / 0.929 / 0.841 |
| x4 | 0.7543 | 0.747 | 0.756 | 0.803 | 0.795 | 0.798 | 10.1 / 10.4 / 9.9 / 8.0 / 10.6 | 81 / 14 / 3 / 2 | 0.903 / 0.829 / 0.939 / 0.841 |

Count of each district per grid in our 800 optimised grids, against grids each advisor labels valid:

| group | Res mean (range) | Ind mean (range) | Com mean (range) | Park mean (range) | Off mean (range) |
|---|---|---|---|---|---|
| our optimised grids | 10.1 (9-11) | 10.3 (4-16) | 10.0 (6-13) | 8.0 (7-9) | 10.6 (4-19) |
| candidate pool, P(valid)>=0.95 | 10.3 (9-11) | 9.6 (4-15) | 8.3 (6-11) | 7.4 (6-9) | 13.3 (7-20) |
| Wellness-valid labelled | 8.4 (2-12) | 8.8 (1-18) | 11.4 (4-20) | 10.5 (7-16) | 9.9 (1-23) |
| Tax-valid labelled | 9.3 (2-16) | 10.7 (2-27) | 13.6 (5-25) | 4.6 (0-9) | 10.8 (0-27) |
| Transportation-valid labelled | 9.7 (5-14) | 8.7 (2-15) | 10.4 (3-18) | 8.9 (3-14) | 11.3 (2-26) |
| Business-valid labelled | 8.4 (3-12) | 12.9 (1-27) | 8.0 (2-14) | 7.1 (0-12) | 12.6 (1-33) |
| all dataset grids | 8.5 (2-14) | 10.5 (2-24) | 11.4 (3-23) | 7.8 (0-14) | 10.8 (1-27) |

Share of each district by ring (0 = outer edge, 3 = centre cell) in our optimised grids:

| ring | Res | Ind | Com | Park | Off |
|---|---|---|---|---|---|
| 0 | 0.19 | 0.22 | 0.22 | 0.15 | 0.23 |
| 1 | 0.22 | 0.20 | 0.19 | 0.18 | 0.20 |
| 2 | 0.22 | 0.20 | 0.19 | 0.18 | 0.20 |
| 3 | 0.22 | 0.21 | 0.18 | 0.18 | 0.20 |

Each advisor's strongest features: our grids against labelled grids

| advisor | feature | Spearman | mean, all labelled | mean, advisor-valid | mean, our grids | range, our grids |
|---|---|---|---|---|---|---|
| Wellness | n Park within 2 of a Com | +0.66 | 6.60 | 8.86 | 7.41 | 4.0-9.0 |
| Wellness | n Park | +0.65 | 7.82 | 10.47 | 7.97 | 7.0-9.0 |
| Wellness | n Park within 2 of a Park | +0.63 | 6.49 | 9.51 | 6.96 | 4.0-9.0 |
| Wellness | n Park touching a Park | +0.61 | 4.25 | 7.02 | 4.91 | 2.0-8.0 |
| Wellness | edge pairs Park-Park | +0.60 | 2.88 | 5.14 | 3.18 | 1.0-6.0 |
| Wellness | mean dist Res->nearest Park | -0.57 | 2.14 | 1.51 | 1.26 | 1.0-1.8 |
| Tax | n Park | -0.69 | 7.91 | 4.63 | 7.97 | 7.0-9.0 |
| Tax | n Park within 2 of a Park | -0.66 | 6.62 | 3.23 | 6.96 | 4.0-9.0 |
| Tax | n Park touching a Park | -0.58 | 4.35 | 1.89 | 4.91 | 2.0-8.0 |
| Tax | edge pairs Park-Park | -0.56 | 2.96 | 1.17 | 3.18 | 1.0-6.0 |
| Tax | n Park within 2 of a Ind | -0.56 | 6.19 | 3.49 | 5.67 | 1.0-8.0 |
| Tax | n Park in ring 0 (0=edge) | -0.56 | 3.83 | 2.09 | 3.48 | 0.0-8.0 |
| Transportation | n Res within 2 of a Com | +0.40 | 7.36 | 8.66 | 10.05 | 8.0-11.0 |
| Transportation | n clusters Res | +0.40 | 5.52 | 6.55 | 8.56 | 5.0-11.0 |
| Transportation | n Park touching a Res | +0.40 | 3.96 | 5.19 | 6.97 | 5.0-9.0 |
| Transportation | n Res touching a Park | +0.40 | 4.05 | 5.28 | 8.34 | 6.0-11.0 |
| Transportation | n Res within 2 of a Park | +0.40 | 6.58 | 8.12 | 9.42 | 7.0-11.0 |
| Transportation | edge pairs Res-Park | +0.39 | 5.26 | 7.06 | 11.62 | 7.0-17.0 |
| Business | n Com | -0.56 | 11.36 | 8.02 | 9.96 | 6.0-13.0 |
| Business | n Com within 2 of a Com | -0.55 | 10.22 | 6.40 | 7.57 | 0.0-12.0 |
| Business | n Com touching a Com | -0.51 | 7.39 | 4.01 | 4.51 | 0.0-11.0 |
| Business | edge pairs Com-Com | -0.50 | 5.56 | 2.64 | 2.68 | 0.0-8.0 |
| Business | diagonal pairs Com-Com | -0.49 | 4.41 | 2.02 | 1.76 | 0.0-5.0 |
| Business | n Com within 2 of a Park | -0.44 | 8.05 | 5.48 | 7.68 | 4.0-12.0 |

Cluster structure (connected same-district regions), our grids against all dataset grids:

| district | clusters, ours | clusters, dataset | largest cluster, ours | largest cluster, dataset |
|---|---|---|---|---|
| Res | 8.6 | 5.5 | 2.1 | 3.0 |
| Ind | 4.9 | 5.6 | 4.7 | 4.2 |
| Com | 7.3 | 6.2 | 2.7 | 4.3 |
| Park | 4.8 | 5.0 | 3.4 | 3.0 |
| Off | 5.9 | 5.4 | 4.0 | 4.6 |

## G. Do independent runs converge to the same grids?

| set | mean distance to nearest grid in same set (cells) | to nearest grid in b2 | to nearest grid in x4 |
|---|---|---|---|
| b1 | 36.5 | 24.5 | 30.3 |
| b2 | 36.6 | - | 30.2 |
| b3 | 36.4 | 24.5 | 30.5 |
| b4 | 36.8 | 23.8 | 30.4 |
| x1 | 36.4 | 30.2 | 30.5 |
| x2 | 36.3 | 30.3 | 30.4 |
| x3 | 36.3 | 30.1 | 30.3 |
| x4 | 36.6 | 30.4 | - |
| S5 (submitted) |  | 24.5 | 30.5 |

