# vision_final

Pytorch Project
1. Preprocessing => 데이터를 정제, 프로젝트 밖에서
```bash
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
```
2. Dataset => dataset[0], dataset[1], ... 전체 데이터를 index로 접근할 수 있게 만들어 줌
   * nn.utils.data.Dataset을 상속 받아야 함
   * __init__, __len__, __getitem__ 를 무조건 overriding 해야함
3. DataLoader => Mini-Batch size 데이터를 가져와야 함. 섞어서 가져와줌, => collate_fn 
4. Loss를 계산하는 metric
5. Optimizer => Adam, SGD, ..,
6. Model => 논문에 있는 모델
   * nn.Module을 상속 받아야 함
   * __init__, (__call__) forward를 무조건 overriding 해야함
7. Training & Validation

추가적으로 할 일 
1. pretrained model load
2. log를 찍어주기  => loss, 정답률 계산

Pytorch-Lightning Project
