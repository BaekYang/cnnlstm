# cnnlstm
Risk Detection through Firearm Recognition Using Deep Learning-Based Object-Human Heterogeneous Graph Extraction

텐서보드로 보는 로그
1. dataset.py코드를 돌린다.
2. train.py코드를 돌린다.
3. http://localhost:6006/를 킨다.
4. 로그 디렉토리를 켜기 위해 아나콘다cmd 가상환경을 하나 더 킨다.
5. tensorboard --logdir =logs를 친다.
6. 6007로컬 호스트를 켜도 된다. 그럴경우
   tensorboard --logdir =logs --port=6007
   쓰고  http://localhost:6007/를 킨다.
8. 1에포크가 돌면 텐서보드에 학습이 찍힌다.

dataset.py: 이미지 데이터셋을 로드하고, 레이블을 반환하는 클래스를 정의합니다.
inference.py: 훈련된 모델을 사용하여 테스트 데이터에서 성능을 평가합니다.
main.py: 학습 및 평가를 실행하는 메인 스크립트입니다.
model.py: CNN-LSTM 모델 구조를 정의합니다.
params.py: 데이터 경로, 하이퍼파라미터, 모델 저장 경로 등을 설정합니다.
train.py: CNN-LSTM 모델을 학습시키고, 학습이 완료되면 모델을 저장합니다.
