main.py는 실험 소스코드, result는 train과 test의 Loss average를 각 epoch마다 표시한 결과입니다.

<데이터셋>
주어진 와인 데이터를 9:1로 나누어 각각 학습, 평가에 사용하였습니다.

<학습>
criterion: MSELoss
optimizer: Adam

실험결과 데이터셋의 크기가 작다보니 어느정도 학습이 진행된 후 overfitting되는 동향을 확인할 수 있었습니다.