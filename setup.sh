#!/bin/bash

# 데이터 소스 디렉토리와 목적지 디렉토리 지정
SOURCE_DIR="data/CelebA/"
TRAIN_DIR="data/CelebA/train/"
TEST_DIR="data/CelebA/test/"

# 디렉토리 생성
mkdir -p "$TRAIN_DIR"
mkdir -p "$TEST_DIR"

# 파일 목록 가져오기
FILES=("$SOURCE_DIR"/*)

# 파일 목록 섞기
shuf -e "${FILES[@]}" -o "${FILES[@]}"

# 분할 지점 계산
SPLIT_POINT=$((${#FILES[@]} * 80 / 100))

# 파일을 train 또는 test 디렉토리로 이동
for ((i = 0; i < SPLIT_POINT; i++)); do
  mv "${FILES[$i]}" "$TRAIN_DIR"
done

for ((i = SPLIT_POINT; i < ${#FILES[@]}; i++)); do
  mv "${FILES[$i]}" "$TEST_DIR"
done
