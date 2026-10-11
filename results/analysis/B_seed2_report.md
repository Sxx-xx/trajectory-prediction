### 스텝별 평균 오차 (m): step 1, 4, 8, 12 / 전체 평균(= ADE)
hotel      constant velocity         0.055  0.189  0.394  0.617  | ADE 0.323
hotel      LSTM heading + rot aug    0.072  0.236  0.487  0.770  | ADE 0.401
hotel      Social (grid pooling)     0.071  0.236  0.476  0.742  | ADE 0.392
all scenes constant velocity         0.045  0.245  0.634  1.141  | ADE 0.520
all scenes LSTM heading + rot aug    0.069  0.292  0.683  1.150  | ADE 0.558
all scenes Social (grid pooling)     0.063  0.273  0.649  1.101  | ADE 0.530

### 방향 전환 각도별  (ADE, m)

hotel
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| slow/undefined | 623 | 0.180 | 0.249 | 0.244 |
| 0-15° | 370 | 0.460 | 0.559 | 0.537 |
| 15-45° | 55 | 0.896 | 0.954 | 0.974 |
| 45°+ | 5 | 1.629 | 1.611 | 1.679 |

all scenes
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| slow/undefined | 11,101 | 0.218 | 0.247 | 0.257 |
| 0-15° | 14,190 | 0.561 | 0.637 | 0.568 |
| 15-45° | 7,028 | 1.003 | 0.952 | 0.968 |
| 45°+ | 1,335 | 1.705 | 1.549 | 1.572 |

### 관측 속력별  (ADE, m)

hotel
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| <0.2 m/s | 580 | 0.148 | 0.222 | 0.213 |
| 0.2-0.8 | 148 | 0.516 | 0.595 | 0.645 |
| 0.8-1.5 | 285 | 0.541 | 0.635 | 0.600 |
| 1.5+ m/s | 40 | 0.577 | 0.623 | 0.571 |

all scenes
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| <0.2 m/s | 9,554 | 0.175 | 0.215 | 0.222 |
| 0.2-0.8 | 12,887 | 0.581 | 0.576 | 0.600 |
| 0.8-1.5 | 10,615 | 0.519 | 0.553 | 0.541 |
| 1.5+ m/s | 598 | 1.393 | 1.513 | 1.239 |

### 이웃 수별 (3 m 안)  (ADE, m)

hotel
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| 0 (alone) | 60 | 0.242 | 0.307 | 0.314 |
| 1-2 | 560 | 0.308 | 0.390 | 0.389 |
| 3-5 | 400 | 0.344 | 0.418 | 0.397 |
| 6+ | 33 | 0.452 | 0.557 | 0.517 |

all scenes
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| 0 (alone) | 758 | 0.468 | 0.543 | 0.508 |
| 1-2 | 4,101 | 0.564 | 0.594 | 0.575 |
| 3-5 | 9,083 | 0.462 | 0.491 | 0.455 |
| 6+ | 19,712 | 0.543 | 0.593 | 0.564 |

### hotel: ADE 차이(모델 - 등속) 의 speed 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: |
| <0.2 m/s | 55% | +0.040 | +0.035 |
| 0.2-0.8 | 14% | +0.011 | +0.018 |
| 0.8-1.5 | 27% | +0.025 | +0.016 |
| 1.5+ m/s | 4% | +0.002 | -0.000 |
| 합 | 100% | +0.078 | +0.069 |

### hotel: ADE 차이(모델 - 등속) 의 turn 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: |
| slow/undefined | 59% | +0.041 | +0.038 |
| 0-15° | 35% | +0.035 | +0.027 |
| 15-45° | 5% | +0.003 | +0.004 |
| 45°+ | 0% | -0.000 | +0.000 |
| 합 | 100% | +0.078 | +0.069 |

### hotel: ADE 차이(모델 - 등속) 의 neighbors 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: |
| 0 (alone) | 6% | +0.004 | +0.004 |
| 1-2 | 53% | +0.043 | +0.043 |
| 3-5 | 38% | +0.028 | +0.020 |
| 6+ | 3% | +0.003 | +0.002 |
| 합 | 100% | +0.078 | +0.069 |

### 마지막 스텝 오차의 진행 방향 성분 (관측 마지막 변위 >= 0.08 m 인 샘플)
along 평균: + 면 예측이 실제보다 앞으로 더 감 / |along|, |cross|: 절댓값 평균 (m)
| 그룹 | n | 모델 | along 평균 | abs(along) | abs(cross) |
| :--- | --: | :--- | --: | --: | --: |
| hotel | 493 | constant velocity | +0.417 | 0.760 | 0.585 |
| hotel | 493 | LSTM heading + rot aug | -0.085 | 0.911 | 0.661 |
| hotel | 493 | Social (grid pooling) | +0.090 | 0.901 | 0.648 |
| all scenes | 23,334 | constant velocity | +0.748 | 1.115 | 0.757 |
| all scenes | 23,334 | LSTM heading + rot aug | -0.047 | 1.024 | 0.892 |
| all scenes | 23,334 | Social (grid pooling) | +0.109 | 1.032 | 0.752 |

### 4.8 s 이동 거리: 평균과, 등속 이동 거리(12 x 마지막 관측 변위)에 대한 회귀 기울기
| 그룹 | n | 대상 | 평균 이동 거리 (m) | 기울기 |
| :--- | --: | :--- | --: | --: |
| hotel | 493 | actual future | 4.381 | 1.08 |
| hotel | 493 | constant velocity | 4.708 | 1.00 |
| hotel | 493 | LSTM heading + rot aug | 4.209 | 0.89 |
| hotel | 493 | Social (grid pooling) | 4.394 | 0.88 |
| all scenes | 23,334 | actual future | 4.778 | 0.71 |
| all scenes | 23,334 | constant velocity | 5.366 | 1.00 |
| all scenes | 23,334 | LSTM heading + rot aug | 4.621 | 0.76 |
| all scenes | 23,334 | Social (grid pooling) | 4.738 | 0.74 |

### 관측 평균 속력 < 0.2 m/s 샘플: 4.8 s 이동 거리 평균 (m)
| 그룹 | n | actual future | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: | --: |
| hotel | 580 | 0.071 | 0.223 | 0.384 | 0.348 |
| all scenes | 9,554 | 0.324 | 0.228 | 0.294 | 0.309 |

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

saved docs/figures/B_seed2_*.png, results/analysis/B_seed2_*.csv
