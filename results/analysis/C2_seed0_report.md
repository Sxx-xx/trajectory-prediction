### 스텝별 평균 오차 (m): step 1, 4, 8, 12 / 전체 평균(= ADE)
hotel      constant velocity         0.055  0.189  0.394  0.617  | ADE 0.323
hotel      Transformer + nbr attention0.068  0.195  0.384  0.585  | ADE 0.316
hotel      Transformer + nbr attn + residual0.056  0.194  0.415  0.656  | ADE 0.339
all scenes constant velocity         0.045  0.245  0.634  1.141  | ADE 0.520
all scenes Transformer + nbr attention0.082  0.304  0.654  1.059  | ADE 0.535
all scenes Transformer + nbr attn + residual0.048  0.241  0.606  1.079  | ADE 0.498

### 방향 전환 각도별  (ADE, m)

hotel
| bin | n | constant velocity | Transformer + nbr attention | Transformer + nbr attn + residual |
| :--- | --: | --: | --: | --: |
| slow/undefined | 623 | 0.180 | 0.162 | 0.227 |
| 0-15° | 370 | 0.460 | 0.479 | 0.449 |
| 15-45° | 55 | 0.896 | 0.842 | 0.762 |
| 45°+ | 5 | 1.629 | 1.614 | 1.484 |

all scenes
| bin | n | constant velocity | Transformer + nbr attention | Transformer + nbr attn + residual |
| :--- | --: | --: | --: | --: |
| slow/undefined | 11,101 | 0.218 | 0.260 | 0.267 |
| 0-15° | 14,190 | 0.561 | 0.604 | 0.535 |
| 15-45° | 7,028 | 1.003 | 0.899 | 0.847 |
| 45°+ | 1,335 | 1.705 | 1.333 | 1.390 |

### 관측 속력별  (ADE, m)

hotel
| bin | n | constant velocity | Transformer + nbr attention | Transformer + nbr attn + residual |
| :--- | --: | --: | --: | --: |
| <0.2 m/s | 580 | 0.148 | 0.143 | 0.200 |
| 0.2-0.8 | 148 | 0.516 | 0.479 | 0.496 |
| 0.8-1.5 | 285 | 0.541 | 0.548 | 0.510 |
| 1.5+ m/s | 40 | 0.577 | 0.571 | 0.555 |

all scenes
| bin | n | constant velocity | Transformer + nbr attention | Transformer + nbr attn + residual |
| :--- | --: | --: | --: | --: |
| <0.2 m/s | 9,554 | 0.175 | 0.228 | 0.233 |
| 0.2-0.8 | 12,887 | 0.581 | 0.612 | 0.570 |
| 0.8-1.5 | 10,615 | 0.519 | 0.528 | 0.504 |
| 1.5+ m/s | 598 | 1.393 | 1.289 | 1.100 |

### 이웃 수별 (3 m 안)  (ADE, m)

hotel
| bin | n | constant velocity | Transformer + nbr attention | Transformer + nbr attn + residual |
| :--- | --: | --: | --: | --: |
| 0 (alone) | 60 | 0.242 | 0.237 | 0.285 |
| 1-2 | 560 | 0.308 | 0.315 | 0.329 |
| 3-5 | 400 | 0.344 | 0.320 | 0.352 |
| 6+ | 33 | 0.452 | 0.431 | 0.449 |

all scenes
| bin | n | constant velocity | Transformer + nbr attention | Transformer + nbr attn + residual |
| :--- | --: | --: | --: | --: |
| 0 (alone) | 758 | 0.468 | 0.577 | 0.487 |
| 1-2 | 4,101 | 0.564 | 0.500 | 0.503 |
| 3-5 | 9,083 | 0.462 | 0.492 | 0.463 |
| 6+ | 19,712 | 0.543 | 0.628 | 0.538 |

### hotel: ADE 차이(모델 - 등속) 의 speed 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | Transformer + nbr attention | Transformer + nbr attn + residual |
| :--- | --: | --: | --: |
| <0.2 m/s | 55% | -0.003 | +0.029 |
| 0.2-0.8 | 14% | -0.005 | -0.003 |
| 0.8-1.5 | 27% | +0.002 | -0.009 |
| 1.5+ m/s | 4% | -0.000 | -0.001 |
| 합 | 100% | -0.007 | +0.016 |

### hotel: ADE 차이(모델 - 등속) 의 turn 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | Transformer + nbr attention | Transformer + nbr attn + residual |
| :--- | --: | --: | --: |
| slow/undefined | 59% | -0.011 | +0.028 |
| 0-15° | 35% | +0.007 | -0.004 |
| 15-45° | 5% | -0.003 | -0.007 |
| 45°+ | 0% | -0.000 | -0.001 |
| 합 | 100% | -0.007 | +0.016 |

### hotel: ADE 차이(모델 - 등속) 의 neighbors 묶음별 기여 (m, 합 = 전체 차이)
| bin | 샘플 비율 | Transformer + nbr attention | Transformer + nbr attn + residual |
| :--- | --: | --: | --: |
| 0 (alone) | 6% | -0.000 | +0.002 |
| 1-2 | 53% | +0.004 | +0.011 |
| 3-5 | 38% | -0.009 | +0.003 |
| 6+ | 3% | -0.001 | -0.000 |
| 합 | 100% | -0.007 | +0.016 |

### 마지막 스텝 오차의 진행 방향 성분 (관측 마지막 변위 >= 0.08 m 인 샘플)
along 평균: + 면 예측이 실제보다 앞으로 더 감 / |along|, |cross|: 절댓값 평균 (m)
| 그룹 | n | 모델 | along 평균 | abs(along) | abs(cross) |
| :--- | --: | :--- | --: | --: | --: |
| hotel | 493 | constant velocity | +0.417 | 0.760 | 0.585 |
| hotel | 493 | Transformer + nbr attention | +0.315 | 0.683 | 0.547 |
| hotel | 493 | Transformer + nbr attn + residual | +0.164 | 0.761 | 0.528 |
| all scenes | 23,334 | constant velocity | +0.748 | 1.115 | 0.757 |
| all scenes | 23,334 | Transformer + nbr attention | -0.062 | 0.917 | 0.786 |
| all scenes | 23,334 | Transformer + nbr attn + residual | +0.342 | 0.990 | 0.715 |

### 4.8 s 이동 거리: 평균과, 등속 이동 거리(12 x 마지막 관측 변위)에 대한 회귀 기울기
| 그룹 | n | 대상 | 평균 이동 거리 (m) | 기울기 |
| :--- | --: | :--- | --: | --: |
| hotel | 493 | actual future | 4.381 | 1.08 |
| hotel | 493 | constant velocity | 4.708 | 1.00 |
| hotel | 493 | Transformer + nbr attention | 4.615 | 1.06 |
| hotel | 493 | Transformer + nbr attn + residual | 4.464 | 0.95 |
| all scenes | 23,334 | actual future | 4.778 | 0.71 |
| all scenes | 23,334 | constant velocity | 5.366 | 1.00 |
| all scenes | 23,334 | Transformer + nbr attention | 4.612 | 0.62 |
| all scenes | 23,334 | Transformer + nbr attn + residual | 4.984 | 0.93 |

### 관측 평균 속력 < 0.2 m/s 샘플: 4.8 s 이동 거리 평균 (m)
| 그룹 | n | actual future | constant velocity | Transformer + nbr attention | Transformer + nbr attn + residual |
| :--- | --: | --: | --: | --: | --: |
| hotel | 580 | 0.071 | 0.223 | 0.241 | 0.363 |
| all scenes | 9,554 | 0.324 | 0.228 | 0.357 | 0.371 |

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

saved docs/figures/C2_seed0_*.png, results/analysis/C2_seed0_*.csv
