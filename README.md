# vision_final

Pytorch Project
1. Preprocessing => 데이터를 정제, 프로젝트 밖에서
```bash
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
```
2. Dataset => dataset[0], dataset[1], ... 전체 데이터를 index로 접근할 수 있게 만들어 줌
3. DataLoader => Mini-Batch size 데이터를 가져와야 함. 섞어서 가져와줌, => collate_fn
4. Optimizer => Adam, SGD, .. 
5. Model => 논문에 있는 모델
5. Training & Validation


Pytorch-Lightning Project
