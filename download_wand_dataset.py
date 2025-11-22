import wandb
import os


WANDB_ENTITY = "ds24m042-fh-technikum-wien" 

WANDB_PROJECT = "pet-breed-classifier-final"

run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type="model-training", reinit=True)

artifact = run.use_artifact(f'{WANDB_ENTITY}/{WANDB_PROJECT}/oxford_pet_dataset:v0', type='dataset')

data_dir = artifact.download(root='1_Model_Training/data_from_wb')


run.finish()