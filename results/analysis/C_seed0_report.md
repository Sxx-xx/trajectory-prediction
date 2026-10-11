### 스텝별 평균 오차 (m): step 1, 4, 8, 12 / 전체 평균(= ADE)
hotel      constant velocity         0.055  0.189  0.394  0.617  | ADE 0.323
hotel      Transformer               0.064  0.222  0.468  0.730  | ADE 0.380
hotel      Transformer + shuffled nbrs0.061  0.211  0.434  0.683  | ADE 0.356
hotel      Transformer + nbr attention0.068  0.195  0.384  0.585  | ADE 0.316
all scenes constant velocity         0.045  0.245  0.634  1.141  | ADE 0.520
all scenes Transformer               0.056  0.264  0.643  1.113  | ADE 0.526
all scenes Transformer + shuffled nbrs0.072  0.300  0.673  1.131  | ADE 0.553
all scenes Transformer + nbr attention0.082  0.304  0.654  1.059  | ADE 0.535

### 방향 전환 각도별  (ADE, m)

hotel
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| slow/undefined | 623 | 0.180 | 0.215 | 0.192 | 0.162 |
| 0-15° | 370 | 0.460 | 0.560 | 0.538 | 0.479 |
| 15-45° | 55 | 0.896 | 0.933 | 0.885 | 0.842 |
| 45°+ | 5 | 1.629 | 1.532 | 1.535 | 1.614 |

all scenes
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| slow/undefined | 11,101 | 0.218 | 0.235 | 0.241 | 0.260 |
| 0-15° | 14,190 | 0.561 | 0.577 | 0.628 | 0.604 |
| 15-45° | 7,028 | 1.003 | 0.970 | 0.972 | 0.899 |
| 45°+ | 1,335 | 1.705 | 1.526 | 1.520 | 1.333 |

### 관측 속력별  (ADE, m)

hotel
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| <0.2 m/s | 580 | 0.148 | 0.189 | 0.167 | 0.143 |
| 0.2-0.8 | 148 | 0.516 | 0.553 | 0.545 | 0.479 |
| 0.8-1.5 | 285 | 0.541 | 0.643 | 0.604 | 0.548 |
| 1.5+ m/s | 40 | 0.577 | 0.638 | 0.631 | 0.571 |

all scenes
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| <0.2 m/s | 9,554 | 0.175 | 0.203 | 0.212 | 0.228 |
| 0.2-0.8 | 12,887 | 0.581 | 0.581 | 0.570 | 0.612 |
| 0.8-1.5 | 10,615 | 0.519 | 0.557 | 0.576 | 0.528 |
| 1.5+ m/s | 598 | 1.393 | 1.238 | 1.414 | 1.289 |

### 이웃 수별 (3 m 안)  (ADE, m)

hotel
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| 0 (alone) | 60 | 0.242 | 0.281 | 0.286 | 0.237 |
| 1-2 | 560 | 0.308 | 0.372 | 0.353 | 0.315 |
| 3-5 | 400 | 0.344 | 0.395 | 0.362 | 0.320 |
| 6+ | 33 | 0.452 | 0.511 | 0.468 | 0.431 |

all scenes
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| 0 (alone) | 758 | 0.468 | 0.508 | 0.553 | 0.577 |
| 1-2 | 4,101 | 0.564 | 0.567 | 0.582 | 0.500 |
| 3-5 | 9,083 | 0.462 | 0.459 | 0.476 | 0.492 |
| 6+ | 19,712 | 0.543 | 0.555 | 0.607 | 0.628 |

### hotel: ADE 차이(모델 - 등속) 의 speed 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: |
| <0.2 m/s | 55% | +0.022 | +0.010 | -0.003 |
| 0.2-0.8 | 14% | +0.005 | +0.004 | -0.005 |
| 0.8-1.5 | 27% | +0.027 | +0.017 | +0.002 |
| 1.5+ m/s | 4% | +0.002 | +0.002 | -0.000 |
| 합 | 100% | +0.057 | +0.033 | -0.007 |

### hotel: ADE 차이(모델 - 등속) 의 turn 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: |
| slow/undefined | 59% | +0.021 | +0.007 | -0.011 |
| 0-15° | 35% | +0.035 | +0.027 | +0.007 |
| 15-45° | 5% | +0.002 | -0.001 | -0.003 |
| 45°+ | 0% | -0.000 | -0.000 | -0.000 |
| 합 | 100% | +0.057 | +0.033 | -0.007 |

### hotel: ADE 차이(모델 - 등속) 의 neighbors 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: |
| 0 (alone) | 6% | +0.002 | +0.002 | -0.000 |
| 1-2 | 53% | +0.034 | +0.024 | +0.004 |
| 3-5 | 38% | +0.019 | +0.007 | -0.009 |
| 6+ | 3% | +0.002 | +0.000 | -0.001 |
| 합 | 100% | +0.057 | +0.033 | -0.007 |

### 마지막 스텝 오차의 진행 방향 성분 (관측 마지막 변위 >= 0.08 m 인 샘플)
along 평균: + 면 예측이 실제보다 앞으로 더 감 / |along|, |cross|: 절댓값 평균 (m)
| 그룹 | n | 모델 | along 평균 | abs(along) | abs(cross) |
| :--- | --: | :--- | --: | --: | --: |
| hotel | 493 | constant velocity | +0.417 | 0.760 | 0.585 |
| hotel | 493 | Transformer | -0.126 | 0.900 | 0.670 |
| hotel | 493 | Transformer + shuffled nbrs | +0.046 | 0.832 | 0.629 |
| hotel | 493 | Transformer + nbr attention | +0.315 | 0.683 | 0.547 |
| all scenes | 23,334 | constant velocity | +0.748 | 1.115 | 0.757 |
| all scenes | 23,334 | Transformer | +0.026 | 1.050 | 0.785 |
| all scenes | 23,334 | Transformer + shuffled nbrs | -0.111 | 1.098 | 0.759 |
| all scenes | 23,334 | Transformer + nbr attention | -0.062 | 0.917 | 0.786 |

### 4.8 s 이동 거리: 평균과, 등속 이동 거리(12 x 마지막 관측 변위)에 대한 회귀 기울기
| 그룹 | n | 대상 | 평균 이동 거리 (m) | 기울기 |
| :--- | --: | :--- | --: | --: |
| hotel | 493 | actual future | 4.381 | 1.08 |
| hotel | 493 | constant velocity | 4.708 | 1.00 |
| hotel | 493 | Transformer | 4.170 | 0.87 |
| hotel | 493 | Transformer + shuffled nbrs | 4.347 | 0.93 |
| hotel | 493 | Transformer + nbr attention | 4.615 | 1.06 |
| all scenes | 23,334 | actual future | 4.778 | 0.71 |
| all scenes | 23,334 | constant velocity | 5.366 | 1.00 |
| all scenes | 23,334 | Transformer | 4.648 | 0.78 |
| all scenes | 23,334 | Transformer + shuffled nbrs | 4.515 | 0.66 |
| all scenes | 23,334 | Transformer + nbr attention | 4.612 | 0.62 |

### 관측 평균 속력 < 0.2 m/s 샘플: 4.8 s 이동 거리 평균 (m)
| 그룹 | n | actual future | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: | --: |
| hotel | 580 | 0.071 | 0.223 | 0.313 | 0.287 | 0.241 |
| all scenes | 9,554 | 0.324 | 0.228 | 0.294 | 0.332 | 0.357 |

### 실제 4.8 s 이동 거리의 등속 이동 거리에 대한 기울기 (마지막 관측 변위 >= 0.08 m)
| 장면 | test 기울기 (n) | train 기울기 (n) |
| :--- | --: | --: |
| eth | 0.61 (133) | 0.86 (21,101) |
| hotel | 1.08 (493) | 0.85 (20,778) |
| univ | 0.77 (17,738) | 0.88 (6,186) |
| zara1 | 0.88 (2,074) | 0.84 (19,419) |
| zara2 | 0.90 (2,896) | 0.84 (18,874) |

### 속도를 k 배로 줄인 등속 모델의 test ADE (m)
| k | eth | hotel | univ | zara1 | zara2 | AVG |
| --: | --: | --: | --: | --: | --: | --: |
| 1.0 | 0.995 | 0.323 | 0.524 | 0.431 | 0.326 | 0.520 |
| 0.9 | 0.933 | 0.331 | 0.507 | 0.483 | 0.341 | 0.519 |
| 0.88 | 0.934 | 0.339 | 0.509 | 0.510 | 0.352 | 0.529 |
| 0.85 | 0.942 | 0.354 | 0.514 | 0.556 | 0.372 | 0.548 |
| 0.8 | 0.972 | 0.384 | 0.531 | 0.644 | 0.411 | 0.588 |

saved docs/figures/C_seed0_*.png, results/analysis/C_seed0_*.csv
