# GabrielleNLP-TextAutoencoder
Text Auto-encoder (Bi-LSTM Seq2Seq) for Document Clustering

- 문자 단위 토크나이저 사용: 적은 수의 텍스트 사이의 클러스터링 성능 확인 목적
- Bi-LSTM 기반 인코더-디코더 구조: Attention Mechanism은 활용하지 않은, 순수하게 압축된 `hidden_state`