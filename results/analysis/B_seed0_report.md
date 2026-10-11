### 스텝별 평균 오차 (m): step 1, 4, 8, 12 / 전체 평균(= ADE)
hotel      constant velocity         0.055  0.189  0.394  0.617  | ADE 0.323
hotel      LSTM heading + rot aug    0.072  0.227  0.462  0.730  | ADE 0.382
hotel      Social (grid pooling)     0.080  0.228  0.448  0.701  | ADE 0.373
all scenes constant velocity         0.045  0.245  0.634  1.141  | ADE 0.520
all scenes LSTM heading + rot aug    0.059  0.270  0.647  1.110  | ADE 0.529
all scenes Social (grid pooling)     0.065  0.274  0.645  1.102  | ADE 0.529

### 방향 전환 각도별  (ADE, m)

hotel
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| slow/undefined | 623 | 0.180 | 0.228 | 0.211 |
| 0-15° | 370 | 0.460 | 0.541 | 0.541 |
| 15-45° | 55 | 0.896 | 0.927 | 0.968 |
| 45°+ | 5 | 1.629 | 1.653 | 1.596 |

all scenes
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| slow/undefined | 11,101 | 0.218 | 0.244 | 0.252 |
| 0-15° | 14,190 | 0.561 | 0.575 | 0.577 |
| 15-45° | 7,028 | 1.003 | 0.975 | 0.952 |
| 45°+ | 1,335 | 1.705 | 1.534 | 1.507 |

### 관측 속력별  (ADE, m)

hotel
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| <0.2 m/s | 580 | 0.148 | 0.197 | 0.182 |
| 0.2-0.8 | 148 | 0.516 | 0.581 | 0.622 |
| 0.8-1.5 | 285 | 0.541 | 0.618 | 0.589 |
| 1.5+ m/s | 40 | 0.577 | 0.633 | 0.686 |

all scenes
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| <0.2 m/s | 9,554 | 0.175 | 0.206 | 0.222 |
| 0.2-0.8 | 12,887 | 0.581 | 0.583 | 0.602 |
| 0.8-1.5 | 10,615 | 0.519 | 0.547 | 0.549 |
| 1.5+ m/s | 598 | 1.393 | 1.287 | 1.206 |

### 이웃 수별 (3 m 안)  (ADE, m)

hotel
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| 0 (alone) | 60 | 0.242 | 0.282 | 0.305 |
| 1-2 | 560 | 0.308 | 0.368 | 0.371 |
| 3-5 | 400 | 0.344 | 0.404 | 0.374 |
| 6+ | 33 | 0.452 | 0.523 | 0.516 |

all scenes
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| 0 (alone) | 758 | 0.468 | 0.514 | 0.504 |
| 1-2 | 4,101 | 0.564 | 0.559 | 0.571 |
| 3-5 | 9,083 | 0.462 | 0.469 | 0.453 |
| 6+ | 19,712 | 0.543 | 0.565 | 0.573 |

### hotel: ADE 차이(모델 - 등속) 의 speed 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: |
| <0.2 m/s | 55% | +0.027 | +0.018 |
| 0.2-0.8 | 14% | +0.009 | +0.015 |
| 0.8-1.5 | 27% | +0.021 | +0.013 |
| 1.5+ m/s | 4% | +0.002 | +0.004 |
| 합 | 100% | +0.059 | +0.050 |

### hotel: ADE 차이(모델 - 등속) 의 turn 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: |
| slow/undefined | 59% | +0.029 | +0.018 |
| 0-15° | 35% | +0.028 | +0.028 |
| 15-45° | 5% | +0.002 | +0.004 |
| 45°+ | 0% | +0.000 | -0.000 |
| 합 | 100% | +0.059 | +0.050 |

### hotel: ADE 차이(모델 - 등속) 의 neighbors 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: |
| 0 (alone) | 6% | +0.002 | +0.004 |
| 1-2 | 53% | +0.032 | +0.033 |
| 3-5 | 38% | +0.023 | +0.011 |
| 6+ | 3% | +0.002 | +0.002 |
| 합 | 100% | +0.059 | +0.050 |

### 마지막 스텝 오차의 진행 방향 성분 (관측 마지막 변위 >= 0.08 m 인 샘플)
along 평균: + 면 예측이 실제보다 앞으로 더 감 / |along|, |cross|: 절댓값 평균 (m)
| 그룹 | n | 모델 | along 평균 | abs(along) | abs(cross) |
| :--- | --: | :--- | --: | --: | --: |
| hotel | 493 | constant velocity | +0.417 | 0.760 | 0.585 |
| hotel | 493 | LSTM heading + rot aug | -0.060 | 0.912 | 0.625 |
| hotel | 493 | Social (grid pooling) | +0.049 | 0.862 | 0.665 |
| all scenes | 23,334 | constant velocity | +0.748 | 1.115 | 0.757 |
| all scenes | 23,334 | LSTM heading + rot aug | +0.059 | 1.036 | 0.784 |
| all scenes | 23,334 | Social (grid pooling) | +0.060 | 1.034 | 0.758 |

### 4.8 s 이동 거리: 평균과, 등속 이동 거리(12 x 마지막 관측 변위)에 대한 회귀 기울기
| 그룹 | n | 대상 | 평균 이동 거리 (m) | 기울기 |
| :--- | --: | :--- | --: | --: |
| hotel | 493 | actual future | 4.381 | 1.08 |
| hotel | 493 | constant velocity | 4.708 | 1.00 |
| hotel | 493 | LSTM heading + rot aug | 4.233 | 0.87 |
| hotel | 493 | Social (grid pooling) | 4.358 | 0.90 |
| all scenes | 23,334 | actual future | 4.778 | 0.71 |
| all scenes | 23,334 | constant velocity | 5.366 | 1.00 |
| all scenes | 23,334 | LSTM heading + rot aug | 4.685 | 0.74 |
| all scenes | 23,334 | Social (grid pooling) | 4.692 | 0.79 |

### 관측 평균 속력 < 0.2 m/s 샘플: 4.8 s 이동 거리 평균 (m)
| 그룹 | n | actual future | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: | --: |
| hotel | 580 | 0.071 | 0.223 | 0.319 | 0.278 |
| all scenes | 9,554 | 0.324 | 0.228 | 0.324 | 0.297 |

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

saved docs/figures/B_seed0_*.png, results/analysis/B_seed0_*.csv
