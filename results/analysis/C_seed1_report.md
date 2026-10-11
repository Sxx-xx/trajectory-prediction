### 스텝별 평균 오차 (m): step 1, 4, 8, 12 / 전체 평균(= ADE)
hotel      constant velocity         0.055  0.189  0.394  0.617  | ADE 0.323
hotel      Transformer               0.067  0.212  0.440  0.698  | ADE 0.361
hotel      Transformer + shuffled nbrs0.068  0.224  0.475  0.747  | ADE 0.389
hotel      Transformer + nbr attention0.063  0.206  0.413  0.636  | ADE 0.340
all scenes constant velocity         0.045  0.245  0.634  1.141  | ADE 0.520
all scenes Transformer               0.066  0.268  0.651  1.129  | ADE 0.535
all scenes Transformer + shuffled nbrs0.066  0.271  0.642  1.100  | ADE 0.527
all scenes Transformer + nbr attention0.084  0.310  0.653  1.055  | ADE 0.536

### 방향 전환 각도별  (ADE, m)

hotel
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| slow/undefined | 623 | 0.180 | 0.174 | 0.246 | 0.219 |
| 0-15° | 370 | 0.460 | 0.571 | 0.542 | 0.457 |
| 15-45° | 55 | 0.896 | 0.956 | 0.875 | 0.823 |
| 45°+ | 5 | 1.629 | 1.590 | 1.579 | 1.505 |

all scenes
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| slow/undefined | 11,101 | 0.218 | 0.236 | 0.248 | 0.258 |
| 0-15° | 14,190 | 0.561 | 0.592 | 0.579 | 0.598 |
| 15-45° | 7,028 | 1.003 | 0.975 | 0.941 | 0.926 |
| 45°+ | 1,335 | 1.705 | 1.575 | 1.492 | 1.403 |

### 관측 속력별  (ADE, m)

hotel
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| <0.2 m/s | 580 | 0.148 | 0.145 | 0.216 | 0.194 |
| 0.2-0.8 | 148 | 0.516 | 0.567 | 0.589 | 0.534 |
| 0.8-1.5 | 285 | 0.541 | 0.652 | 0.605 | 0.507 |
| 1.5+ m/s | 40 | 0.577 | 0.652 | 0.616 | 0.547 |

all scenes
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| <0.2 m/s | 9,554 | 0.175 | 0.203 | 0.213 | 0.231 |
| 0.2-0.8 | 12,887 | 0.581 | 0.590 | 0.599 | 0.587 |
| 0.8-1.5 | 10,615 | 0.519 | 0.560 | 0.557 | 0.506 |
| 1.5+ m/s | 598 | 1.393 | 1.294 | 1.189 | 1.408 |

### 이웃 수별 (3 m 안)  (ADE, m)

hotel
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| 0 (alone) | 60 | 0.242 | 0.247 | 0.299 | 0.297 |
| 1-2 | 560 | 0.308 | 0.356 | 0.385 | 0.339 |
| 3-5 | 400 | 0.344 | 0.375 | 0.399 | 0.335 |
| 6+ | 33 | 0.452 | 0.482 | 0.491 | 0.497 |

all scenes
| bin | n | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: |
| 0 (alone) | 758 | 0.468 | 0.521 | 0.528 | 0.555 |
| 1-2 | 4,101 | 0.564 | 0.568 | 0.573 | 0.534 |
| 3-5 | 9,083 | 0.462 | 0.467 | 0.454 | 0.482 |
| 6+ | 19,712 | 0.543 | 0.577 | 0.552 | 0.602 |

### hotel: ADE 차이(모델 - 등속) 의 speed 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: |
| <0.2 m/s | 55% | -0.002 | +0.037 | +0.025 |
| 0.2-0.8 | 14% | +0.007 | +0.010 | +0.003 |
| 0.8-1.5 | 27% | +0.030 | +0.017 | -0.009 |
| 1.5+ m/s | 4% | +0.003 | +0.002 | -0.001 |
| 합 | 100% | +0.038 | +0.066 | +0.017 |

### hotel: ADE 차이(모델 - 등속) 의 turn 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: |
| slow/undefined | 59% | -0.004 | +0.039 | +0.023 |
| 0-15° | 35% | +0.039 | +0.029 | -0.001 |
| 15-45° | 5% | +0.003 | -0.001 | -0.004 |
| 45°+ | 0% | -0.000 | -0.000 | -0.001 |
| 합 | 100% | +0.038 | +0.066 | +0.017 |

### hotel: ADE 차이(모델 - 등속) 의 neighbors 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: |
| 0 (alone) | 6% | +0.000 | +0.003 | +0.003 |
| 1-2 | 53% | +0.025 | +0.041 | +0.016 |
| 3-5 | 38% | +0.012 | +0.021 | -0.004 |
| 6+ | 3% | +0.001 | +0.001 | +0.001 |
| 합 | 100% | +0.038 | +0.066 | +0.017 |

### 마지막 스텝 오차의 진행 방향 성분 (관측 마지막 변위 >= 0.08 m 인 샘플)
along 평균: + 면 예측이 실제보다 앞으로 더 감 / |along|, |cross|: 절댓값 평균 (m)
| 그룹 | n | 모델 | along 평균 | abs(along) | abs(cross) |
| :--- | --: | :--- | --: | --: | --: |
| hotel | 493 | constant velocity | +0.417 | 0.760 | 0.585 |
| hotel | 493 | Transformer | -0.120 | 0.882 | 0.694 |
| hotel | 493 | Transformer + shuffled nbrs | -0.044 | 0.906 | 0.589 |
| hotel | 493 | Transformer + nbr attention | +0.155 | 0.731 | 0.555 |
| all scenes | 23,334 | constant velocity | +0.748 | 1.115 | 0.757 |
| all scenes | 23,334 | Transformer | +0.014 | 1.065 | 0.795 |
| all scenes | 23,334 | Transformer + shuffled nbrs | +0.035 | 1.033 | 0.769 |
| all scenes | 23,334 | Transformer + nbr attention | -0.047 | 0.968 | 0.724 |

### 4.8 s 이동 거리: 평균과, 등속 이동 거리(12 x 마지막 관측 변위)에 대한 회귀 기울기
| 그룹 | n | 대상 | 평균 이동 거리 (m) | 기울기 |
| :--- | --: | :--- | --: | --: |
| hotel | 493 | actual future | 4.381 | 1.08 |
| hotel | 493 | constant velocity | 4.708 | 1.00 |
| hotel | 493 | Transformer | 4.178 | 0.88 |
| hotel | 493 | Transformer + shuffled nbrs | 4.252 | 0.90 |
| hotel | 493 | Transformer + nbr attention | 4.478 | 0.97 |
| all scenes | 23,334 | actual future | 4.778 | 0.71 |
| all scenes | 23,334 | constant velocity | 5.366 | 1.00 |
| all scenes | 23,334 | Transformer | 4.636 | 0.79 |
| all scenes | 23,334 | Transformer + shuffled nbrs | 4.666 | 0.79 |
| all scenes | 23,334 | Transformer + nbr attention | 4.602 | 0.64 |

### 관측 평균 속력 < 0.2 m/s 샘플: 4.8 s 이동 거리 평균 (m)
| 그룹 | n | actual future | constant velocity | Transformer | Transformer + shuffled nbrs | Transformer + nbr attention |
| :--- | --: | --: | --: | --: | --: | --: |
| hotel | 580 | 0.071 | 0.223 | 0.234 | 0.385 | 0.338 |
| all scenes | 9,554 | 0.324 | 0.228 | 0.282 | 0.305 | 0.367 |

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

saved docs/figures/C_seed1_*.png, results/analysis/C_seed1_*.csv
