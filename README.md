# vision_final 
Author: Seungyoon Paik

Model Retrieved From: https://github.com/kylemin/S3D

Training Dataset Retrieved From: http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar

Referenced: 

https://lightning.ai/pages/open-source/

https://github.com/features/copilot


## Getting Start
### Docker
```bash
docker build -t vision .
export WANDB_API_KEY=[MY_API_KEY]
docker-compose up -d

```

### Local
```bash
pip install -r requirements.txt
export PYTHONPATH={ABSOLUTE_PATH_TO_PROJECT}
wandb login [MY_API_KEY]
python src/main.py
```




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
3. Training
4. logging using wandb
   * Loss : training, validation
     * entire
   * Accuracy : training, validation (torchmetrics)
     * entire
     * each class
```bash
export WANDB_API_KEY=[MY_API_KEY]
```
5. Data augmentation
6. Save Model
6. Dockerfile & Docker-compose