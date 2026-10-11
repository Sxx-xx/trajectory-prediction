### null 가중치와 "가장 많이 본 이웃 = 가장 가까운 이웃" 비율 (이웃 2명 이상)
| 모델 | 장면 | null 가중치 | top = nearest | 무작위 기대값 |
| :--- | :--- | --: | --: | --: |
| tf_none_heading_social_seed0 | eth | 0.295 | 0.044 | 0.176 |
| tf_none_heading_social_seed0 | hotel | 0.213 | 0.167 | 0.152 |
| tf_none_heading_social_seed0 | univ | 0.087 | 0.004 | 0.033 |
| tf_none_heading_social_seed0 | zara1 | 0.278 | 0.259 | 0.182 |
| tf_none_heading_social_seed0 | zara2 | 0.161 | 0.075 | 0.114 |
| tf_none_heading_social_seed0 | all scenes | 0.207 | 0.110 | 0.131 |
| tf_none_heading_social_seed1 | eth | 0.225 | 0.044 | 0.176 |
| tf_none_heading_social_seed1 | hotel | 0.333 | 0.249 | 0.152 |
| tf_none_heading_social_seed1 | univ | 0.039 | 0.012 | 0.033 |
| tf_none_heading_social_seed1 | zara1 | 0.266 | 0.217 | 0.182 |
| tf_none_heading_social_seed1 | zara2 | 0.166 | 0.138 | 0.114 |
| tf_none_heading_social_seed1 | all scenes | 0.206 | 0.132 | 0.131 |
| tf_none_heading_social_seed2 | eth | 0.352 | 0.044 | 0.176 |
| tf_none_heading_social_seed2 | hotel | 0.299 | 0.227 | 0.152 |
| tf_none_heading_social_seed2 | univ | 0.108 | 0.003 | 0.033 |
| tf_none_heading_social_seed2 | zara1 | 0.288 | 0.221 | 0.182 |
| tf_none_heading_social_seed2 | zara2 | 0.172 | 0.076 | 0.114 |
| tf_none_heading_social_seed2 | all scenes | 0.244 | 0.114 | 0.131 |
| tf_none_heading_social_shuffle_seed0 | eth | 0.228 | 0.039 | 0.176 |
| tf_none_heading_social_shuffle_seed0 | hotel | 0.306 | 0.133 | 0.152 |
| tf_none_heading_social_shuffle_seed0 | univ | 0.026 | 0.008 | 0.033 |
| tf_none_heading_social_shuffle_seed0 | zara1 | 0.372 | 0.161 | 0.182 |
| tf_none_heading_social_shuffle_seed0 | zara2 | 0.106 | 0.088 | 0.114 |
| tf_none_heading_social_shuffle_seed0 | all scenes | 0.207 | 0.086 | 0.131 |
| tf_none_heading_social_shuffle_seed1 | eth | 0.176 | 0.050 | 0.176 |
| tf_none_heading_social_shuffle_seed1 | hotel | 0.173 | 0.126 | 0.152 |
| tf_none_heading_social_shuffle_seed1 | univ | 0.060 | 0.011 | 0.033 |
| tf_none_heading_social_shuffle_seed1 | zara1 | 0.317 | 0.135 | 0.182 |
| tf_none_heading_social_shuffle_seed1 | zara2 | 0.137 | 0.080 | 0.114 |
| tf_none_heading_social_shuffle_seed1 | all scenes | 0.173 | 0.080 | 0.131 |
| tf_none_heading_social_shuffle_seed2 | eth | 0.409 | 0.028 | 0.176 |
| tf_none_heading_social_shuffle_seed2 | hotel | 0.368 | 0.123 | 0.152 |
| tf_none_heading_social_shuffle_seed2 | univ | 0.023 | 0.007 | 0.033 |
| tf_none_heading_social_shuffle_seed2 | zara1 | 0.341 | 0.113 | 0.182 |
| tf_none_heading_social_shuffle_seed2 | zara2 | 0.176 | 0.079 | 0.114 |
| tf_none_heading_social_shuffle_seed2 | all scenes | 0.263 | 0.070 | 0.131 |

### 이웃 한 명당 상대 가중치 (1 = 균등하게 나눌 때). all scenes = 장면 평균, 괄호는 이웃 수 합
| 모델 | dist <1 m | dist 1-2 m | dist 2-4 m | dist 4+ m | approaching | receding | ahead | behind |
| :--- | --: | --: | --: | --: | --: | --: | --: | --: |
| tf_none_heading_social_seed0 | 0.86 (47,746) | 0.84 (83,667) | 0.93 (234,747) | 1.08 (477,065) | 1.02 (347,313) | 1.00 (346,400) | 1.06 (434,390) | 0.95 (408,835) |
| tf_none_heading_social_seed1 | 0.89 (47,746) | 0.85 (83,667) | 0.93 (234,747) | 1.07 (477,065) | 1.00 (347,313) | 1.01 (346,400) | 1.02 (434,390) | 1.00 (408,835) |
| tf_none_heading_social_seed2 | 0.82 (47,746) | 0.82 (83,667) | 0.92 (234,747) | 1.10 (477,065) | 1.03 (347,313) | 1.00 (346,400) | 1.07 (434,390) | 0.92 (408,835) |
| tf_none_heading_social_shuffle_seed0 | 0.78 (47,746) | 0.84 (83,667) | 0.93 (234,747) | 1.09 (477,065) | 1.03 (348,441) | 1.00 (368,455) | 1.04 (417,836) | 0.97 (425,389) |
| tf_none_heading_social_shuffle_seed1 | 0.79 (47,746) | 0.86 (83,667) | 0.94 (234,747) | 1.09 (477,065) | 0.94 (349,527) | 1.13 (366,622) | 0.93 (419,390) | 1.07 (423,835) |
| tf_none_heading_social_shuffle_seed2 | 0.65 (47,746) | 0.78 (83,667) | 0.93 (234,747) | 1.13 (477,065) | 1.04 (348,745) | 1.03 (367,934) | 1.06 (416,756) | 0.95 (426,469) |

### 이웃 지우기 시험 (평가 때만, 마지막 관측 프레임). ADE / FDE (m), all scenes = 장면 평균
| 조건 | tf_none_heading_social_seed0 hotel | tf_none_heading_social_seed0 all | tf_none_heading_social_seed1 hotel | tf_none_heading_social_seed1 all | tf_none_heading_social_seed2 hotel | tf_none_heading_social_seed2 all |
| :--- | --: | --: | --: | --: | --: | --: |
| all neighbors | 0.316 / 0.585 | 0.535 / 1.059 | 0.340 / 0.636 | 0.536 / 1.055 | 0.342 / 0.639 | 0.541 / 1.065 |
| drop <2 m | 0.321 / 0.595 | 0.547 / 1.084 | 0.350 / 0.656 | 0.548 / 1.074 | 0.355 / 0.663 | 0.553 / 1.086 |
| drop 2-4 m | 0.316 / 0.587 | 0.540 / 1.073 | 0.353 / 0.664 | 0.540 / 1.064 | 0.351 / 0.657 | 0.546 / 1.076 |
| drop 4+ m | 0.322 / 0.600 | 0.555 / 1.120 | 0.337 / 0.629 | 0.543 / 1.094 | 0.335 / 0.627 | 0.546 / 1.108 |
| drop all | 0.372 / 0.706 | 0.676 / 1.398 | 0.415 / 0.802 | 0.585 / 1.195 | 0.400 / 0.771 | 0.606 / 1.260 |
| keep positions, zero velocities | 0.323 / 0.614 | 0.542 / 1.101 | 0.368 / 0.717 | 0.534 / 1.097 | 0.352 / 0.676 | 0.524 / 1.080 |

saved docs/figures/C_attention_*.png
