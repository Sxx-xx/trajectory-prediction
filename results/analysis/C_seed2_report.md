### 스텝별 평균 오차 (m): step 1, 4, 8, 12 / 전체 평균(= ADE)
hotel      constant velocity         0.055  0.189  0.394  0.617  | ADE 0.323
hotel      Transformer               0.063  0.203  0.437  0.706  | ADE 0.358
hotel      Transformer + shuffled nbrs0.059  0.204  0.413  0.653  | ADE 0.341
hotel      Transformer + nbr attention0.066  0.208  0.414  0.639  | ADE 0.342
all scenes constant velocity         0.045  0.245  0.634  1.141  | ADE 0.520
all scenes Transformer               0.060  0.264  0.644  1.110  | ADE 0.526
all scenes Transformer + shuffled nbrs0.074  0.300  0.668  1.110  | ADE 0.550
all scenes Transformer + nbr attention0.083  0.302  0.663  1.065  | ADE 0.541

### 방향 전환 각도별  (ADE, m)

hotel
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| slow/undefined | 623 | 0.180 | 0.151 | 0.187 | 0.210 |
| 0-15° | 370 | 0.460 | 0.598 | 0.512 | 0.482 |
| 15-45° | 55 | 0.896 | 0.981 | 0.839 | 0.794 |
| 45°+ | 5 | 1.629 | 1.530 | 1.486 | 1.549 |

all scenes
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| slow/undefined | 11,101 | 0.218 | 0.214 | 0.232 | 0.278 |
| 0-15° | 14,190 | 0.561 | 0.591 | 0.636 | 0.603 |
| 15-45° | 7,028 | 1.003 | 0.971 | 0.949 | 0.878 |
| 45°+ | 1,335 | 1.705 | 1.566 | 1.470 | 1.458 |

### 관측 속력별  (ADE, m)

hotel
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| <0.2 m/s | 580 | 0.148 | 0.129 | 0.164 | 0.191 |
| 0.2-0.8 | 148 | 0.516 | 0.553 | 0.529 | 0.498 |
| 0.8-1.5 | 285 | 0.541 | 0.678 | 0.571 | 0.538 |
| 1.5+ m/s | 40 | 0.577 | 0.690 | 0.585 | 0.569 |

all scenes
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| <0.2 m/s | 9,554 | 0.175 | 0.182 | 0.204 | 0.250 |
| 0.2-0.8 | 12,887 | 0.581 | 0.586 | 0.568 | 0.607 |
| 0.8-1.5 | 10,615 | 0.519 | 0.565 | 0.551 | 0.525 |
| 1.5+ m/s | 598 | 1.393 | 1.269 | 1.494 | 1.303 |

### 이웃 수별 (3 m 안)  (ADE, m)

hotel
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| 0 (alone) | 60 | 0.242 | 0.247 | 0.274 | 0.303 |
| 1-2 | 560 | 0.308 | 0.359 | 0.338 | 0.345 |
| 3-5 | 400 | 0.344 | 0.367 | 0.346 | 0.334 |
| 6+ | 33 | 0.452 | 0.444 | 0.455 | 0.480 |

all scenes
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| 0 (alone) | 758 | 0.468 | 0.512 | 0.555 | 0.642 |
| 1-2 | 4,101 | 0.564 | 0.564 | 0.560 | 0.536 |
| 3-5 | 9,083 | 0.462 | 0.458 | 0.479 | 0.472 |
| 6+ | 19,712 | 0.543 | 0.561 | 0.623 | 0.610 |

### hotel: ADE 차이(모델 - 등속) 의 speed 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: |
| <0.2 m/s | 55% | -0.011 | +0.008 | +0.023 |
| 0.2-0.8 | 14% | +0.005 | +0.002 | -0.003 |
| 0.8-1.5 | 27% | +0.037 | +0.008 | -0.001 |
| 1.5+ m/s | 4% | +0.004 | +0.000 | -0.000 |
| 합 | 100% | +0.036 | +0.018 | +0.020 |

### hotel: ADE 차이(모델 - 등속) 의 turn 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: |
| slow/undefined | 59% | -0.017 | +0.004 | +0.018 |
| 0-15° | 35% | +0.049 | +0.018 | +0.008 |
| 15-45° | 5% | +0.004 | -0.003 | -0.005 |
| 45°+ | 0% | -0.000 | -0.001 | -0.000 |
| 합 | 100% | +0.036 | +0.018 | +0.020 |

### hotel: ADE 차이(모델 - 등속) 의 neighbors 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: |
| 0 (alone) | 6% | +0.000 | +0.002 | +0.003 |
| 1-2 | 53% | +0.027 | +0.016 | +0.020 |
| 3-5 | 38% | +0.009 | +0.001 | -0.004 |
| 6+ | 3% | -0.000 | +0.000 | +0.001 |
| 합 | 100% | +0.036 | +0.018 | +0.020 |

### 마지막 스텝 오차의 진행 방향 성분 (관측 마지막 변위 >= 0.08 m 인 샘플)
along 평균: + 면 예측이 실제보다 앞으로 더 감 / |along|, |cross|: 절댓값 평균 (m)
| 그룹 | n | 모델 | along 평균 | abs(along) | abs(cross) |
| :--- | --: | :--- | --: | --: | --: |
| hotel | 493 | constant velocity | +0.417 | 0.760 | 0.585 |
| hotel | 493 | Transformer | -0.226 | 0.902 | 0.706 |
| hotel | 493 | Transformer + shuffled nbrs | +0.014 | 0.806 | 0.584 |
| hotel | 493 | Transformer + nbr attention | +0.161 | 0.716 | 0.552 |
| all scenes | 23,334 | constant velocity | +0.748 | 1.115 | 0.757 |
| all scenes | 23,334 | Transformer | -0.064 | 1.062 | 0.788 |
| all scenes | 23,334 | Transformer + shuffled nbrs | -0.183 | 1.047 | 0.787 |
| all scenes | 23,334 | Transformer + nbr attention | -0.038 | 0.940 | 0.785 |

### 4.8 s 이동 거리: 평균과, 등속 이동 거리(12 x 마지막 관측 변위)에 대한 회귀 기울기
| 그룹 | n | 대상 | 평균 이동 거리 (m) | 기울기 |
| :--- | --: | :--- | --: | --: |
| hotel | 493 | actual future | 4.381 | 1.08 |
| hotel | 493 | constant velocity | 4.708 | 1.00 |
| hotel | 493 | Transformer | 4.075 | 0.90 |
| hotel | 493 | Transformer + shuffled nbrs | 4.319 | 0.94 |
| hotel | 493 | Transformer + nbr attention | 4.486 | 1.02 |
| all scenes | 23,334 | actual future | 4.778 | 0.71 |
| all scenes | 23,334 | constant velocity | 5.366 | 1.00 |
| all scenes | 23,334 | Transformer | 4.560 | 0.73 |
| all scenes | 23,334 | Transformer + shuffled nbrs | 4.457 | 0.67 |
| all scenes | 23,334 | Transformer + nbr attention | 4.638 | 0.59 |

### 관측 평균 속력 < 0.2 m/s 샘플: 4.8 s 이동 거리 평균 (m)
| 그룹 | n | actual future | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: | --: |
| hotel | 580 | 0.071 | 0.223 | 0.215 | 0.290 | 0.342 |
| all scenes | 9,554 | 0.324 | 0.228 | 0.254 | 0.287 | 0.390 |

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

saved docs/figures/C_seed2_*.png, results/analysis/C_seed2_*.csv
