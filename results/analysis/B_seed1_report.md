### 스텝별 평균 오차 (m): step 1, 4, 8, 12 / 전체 평균(= ADE)
hotel      constant velocity         0.055  0.189  0.394  0.617  | ADE 0.323
hotel      LSTM heading + rot aug    0.074  0.237  0.477  0.744  | ADE 0.393
hotel      Social (grid pooling)     0.071  0.231  0.478  0.773  | ADE 0.397
all scenes constant velocity         0.045  0.245  0.634  1.141  | ADE 0.520
all scenes LSTM heading + rot aug    0.067  0.282  0.663  1.128  | ADE 0.544
all scenes Social (grid pooling)     0.064  0.273  0.647  1.117  | ADE 0.533

### 방향 전환 각도별  (ADE, m)

hotel
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| slow/undefined | 623 | 0.180 | 0.232 | 0.245 |
| 0-15° | 370 | 0.460 | 0.559 | 0.543 |
| 15-45° | 55 | 0.896 | 0.980 | 1.008 |
| 45°+ | 5 | 1.629 | 1.723 | 1.740 |

all scenes
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| slow/undefined | 11,101 | 0.218 | 0.241 | 0.263 |
| 0-15° | 14,190 | 0.561 | 0.604 | 0.568 |
| 15-45° | 7,028 | 1.003 | 0.986 | 0.970 |
| 45°+ | 1,335 | 1.705 | 1.542 | 1.579 |

### 관측 속력별  (ADE, m)

hotel
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| <0.2 m/s | 580 | 0.148 | 0.205 | 0.214 |
| 0.2-0.8 | 148 | 0.516 | 0.603 | 0.655 |
| 0.8-1.5 | 285 | 0.541 | 0.631 | 0.607 |
| 1.5+ m/s | 40 | 0.577 | 0.650 | 0.596 |

all scenes
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| <0.2 m/s | 9,554 | 0.175 | 0.207 | 0.230 |
| 0.2-0.8 | 12,887 | 0.581 | 0.587 | 0.612 |
| 0.8-1.5 | 10,615 | 0.519 | 0.581 | 0.542 |
| 1.5+ m/s | 598 | 1.393 | 1.297 | 1.217 |

### 이웃 수별 (3 m 안)  (ADE, m)

hotel
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| 0 (alone) | 60 | 0.242 | 0.294 | 0.297 |
| 1-2 | 560 | 0.308 | 0.383 | 0.388 |
| 3-5 | 400 | 0.344 | 0.408 | 0.407 |
| 6+ | 33 | 0.452 | 0.557 | 0.601 |

all scenes
| bin | n | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: |
| 0 (alone) | 758 | 0.468 | 0.524 | 0.505 |
| 1-2 | 4,101 | 0.564 | 0.591 | 0.580 |
| 3-5 | 9,083 | 0.462 | 0.484 | 0.459 |
| 6+ | 19,712 | 0.543 | 0.555 | 0.564 |

### hotel: ADE 차이(모델 - 등속) 의 speed 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: |
| <0.2 m/s | 55% | +0.031 | +0.036 |
| 0.2-0.8 | 14% | +0.012 | +0.020 |
| 0.8-1.5 | 27% | +0.024 | +0.018 |
| 1.5+ m/s | 4% | +0.003 | +0.001 |
| 합 | 100% | +0.070 | +0.074 |

### hotel: ADE 차이(모델 - 등속) 의 turn 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: |
| slow/undefined | 59% | +0.031 | +0.039 |
| 0-15° | 35% | +0.035 | +0.029 |
| 15-45° | 5% | +0.004 | +0.006 |
| 45°+ | 0% | +0.000 | +0.001 |
| 합 | 100% | +0.070 | +0.074 |

### hotel: ADE 차이(모델 - 등속) 의 neighbors 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: |
| 0 (alone) | 6% | +0.003 | +0.003 |
| 1-2 | 53% | +0.040 | +0.042 |
| 3-5 | 38% | +0.024 | +0.024 |
| 6+ | 3% | +0.003 | +0.005 |
| 합 | 100% | +0.070 | +0.074 |

### 마지막 스텝 오차의 진행 방향 성분 (관측 마지막 변위 >= 0.08 m 인 샘플)
along 평균: + 면 예측이 실제보다 앞으로 더 감 / |along|, |cross|: 절댓값 평균 (m)
| 그룹 | n | 모델 | along 평균 | abs(along) | abs(cross) |
| :--- | --: | :--- | --: | --: | --: |
| hotel | 493 | constant velocity | +0.417 | 0.760 | 0.585 |
| hotel | 493 | LSTM heading + rot aug | +0.049 | 0.868 | 0.676 |
| hotel | 493 | Social (grid pooling) | +0.117 | 0.908 | 0.676 |
| all scenes | 23,334 | constant velocity | +0.748 | 1.115 | 0.757 |
| all scenes | 23,334 | LSTM heading + rot aug | +0.020 | 1.054 | 0.806 |
| all scenes | 23,334 | Social (grid pooling) | +0.087 | 1.054 | 0.764 |

### 4.8 s 이동 거리: 평균과, 등속 이동 거리(12 x 마지막 관측 변위)에 대한 회귀 기울기
| 그룹 | n | 대상 | 평균 이동 거리 (m) | 기울기 |
| :--- | --: | :--- | --: | --: |
| hotel | 493 | actual future | 4.381 | 1.08 |
| hotel | 493 | constant velocity | 4.708 | 1.00 |
| hotel | 493 | LSTM heading + rot aug | 4.346 | 0.93 |
| hotel | 493 | Social (grid pooling) | 4.419 | 0.86 |
| all scenes | 23,334 | actual future | 4.778 | 0.71 |
| all scenes | 23,334 | constant velocity | 5.366 | 1.00 |
| all scenes | 23,334 | LSTM heading + rot aug | 4.650 | 0.78 |
| all scenes | 23,334 | Social (grid pooling) | 4.713 | 0.81 |

### 관측 평균 속력 < 0.2 m/s 샘플: 4.8 s 이동 거리 평균 (m)
| 그룹 | n | actual future | constant velocity | LSTM heading + rot aug | Social (grid pooling) |
| :--- | --: | --: | --: | --: | --: |
| hotel | 580 | 0.071 | 0.223 | 0.344 | 0.389 |
| all scenes | 9,554 | 0.324 | 0.228 | 0.314 | 0.294 |

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

saved docs/figures/B_seed1_*.png, results/analysis/B_seed1_*.csv
