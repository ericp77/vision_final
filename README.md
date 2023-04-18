# vision_final

Pytorch Project
1. Preprocessing => clean the data from outsied the project 
```bash
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
```
2. Dataset => dataset[0], dataset[1], ... index data for loading and accessing
   * nn.utils.data.Dataset => inherit
   * __init__, __len__, __getitem__   must override
3. DataLoader => Mini-Batch size    load the data and shuffle => collate_fn 
4. Loss를 계산하는 metric
5. Optimizer => Adam, SGD, ..,
6. Model => the model in the paper 
      * nn.Module => inherit 
   * __init__, (__call__) must override forward
7. Training & Validation

Additional Tasks
1. pretrained model load
2. log using wandb  => loss, accuracy  

Pytorch-Lightning Project
1. DataModule
2. LightningModule 
3. Traning
