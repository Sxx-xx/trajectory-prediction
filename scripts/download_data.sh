#!/usr/bin/env bash
# ETH/UCY 보행자 궤적 데이터(5개 장면, leave-one-out 분할)를 data/raw 에 받는다.
#
# 출처: Trajectron++ 저장소의 experiments/pedestrians/raw (MIT 라이선스)
#   https://github.com/StanfordASL/Trajectron-plus-plus
# 이 파일들은 Social-GAN 이 배포한 전처리 데이터(datasets.zip)와 바이트 단위로 같다.
# 재현성을 위해 특정 커밋에 고정해서 받는다.
#
# 사용: bash scripts/download_data.sh

set -euo pipefail

COMMIT=1031c7bd1a444273af378c1ec1dcca907ba59830
BASE="https://raw.githubusercontent.com/StanfordASL/Trajectron-plus-plus/${COMMIT}/experiments/pedestrians/raw"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/data/raw"

# 장면/분할/파일. train, val 은 나머지 4개 장면에서, test 는 해당 장면에서 온다.
FILES=$(cat <<'EOF'
eth/train/biwi_hotel_train.txt
eth/train/crowds_zara01_train.txt
eth/train/crowds_zara02_train.txt
eth/train/crowds_zara03_train.txt
eth/train/students001_train.txt
eth/train/students003_train.txt
eth/train/uni_examples_train.txt
eth/val/biwi_hotel_val.txt
eth/val/crowds_zara01_val.txt
eth/val/crowds_zara02_val.txt
eth/val/crowds_zara03_val.txt
eth/val/students001_val.txt
eth/val/students003_val.txt
eth/val/uni_examples_val.txt
eth/test/biwi_eth.txt
hotel/train/biwi_eth_train.txt
hotel/train/crowds_zara01_train.txt
hotel/train/crowds_zara02_train.txt
hotel/train/crowds_zara03_train.txt
hotel/train/students001_train.txt
hotel/train/students003_train.txt
hotel/train/uni_examples_train.txt
hotel/val/biwi_eth_val.txt
hotel/val/crowds_zara01_val.txt
hotel/val/crowds_zara02_val.txt
hotel/val/crowds_zara03_val.txt
hotel/val/students001_val.txt
hotel/val/students003_val.txt
hotel/val/uni_examples_val.txt
hotel/test/biwi_hotel.txt
univ/train/biwi_eth_train.txt
univ/train/biwi_hotel_train.txt
univ/train/crowds_zara01_train.txt
univ/train/crowds_zara02_train.txt
univ/train/crowds_zara03_train.txt
univ/train/uni_examples_train.txt
univ/val/biwi_eth_val.txt
univ/val/biwi_hotel_val.txt
univ/val/crowds_zara01_val.txt
univ/val/crowds_zara02_val.txt
univ/val/crowds_zara03_val.txt
univ/val/uni_examples_val.txt
univ/test/students001.txt
univ/test/students003.txt
zara1/train/biwi_eth_train.txt
zara1/train/biwi_hotel_train.txt
zara1/train/crowds_zara02_train.txt
zara1/train/crowds_zara03_train.txt
zara1/train/students001_train.txt
zara1/train/students003_train.txt
zara1/train/uni_examples_train.txt
zara1/val/biwi_eth_val.txt
zara1/val/biwi_hotel_val.txt
zara1/val/crowds_zara02_val.txt
zara1/val/crowds_zara03_val.txt
zara1/val/students001_val.txt
zara1/val/students003_val.txt
zara1/val/uni_examples_val.txt
zara1/test/crowds_zara01.txt
zara2/train/biwi_eth_train.txt
zara2/train/biwi_hotel_train.txt
zara2/train/crowds_zara01_train.txt
zara2/train/crowds_zara03_train.txt
zara2/train/students001_train.txt
zara2/train/students003_train.txt
zara2/train/uni_examples_train.txt
zara2/val/biwi_eth_val.txt
zara2/val/biwi_hotel_val.txt
zara2/val/crowds_zara01_val.txt
zara2/val/crowds_zara03_val.txt
zara2/val/students001_val.txt
zara2/val/students003_val.txt
zara2/val/uni_examples_val.txt
zara2/test/crowds_zara02.txt
EOF
)

count=0
for f in ${FILES}; do
  mkdir -p "${DEST}/$(dirname "${f}")"
  curl -sSfL --retry 3 -o "${DEST}/${f}" "${BASE}/${f}"
  count=$((count + 1))
done
echo "${count} files -> ${DEST}"
