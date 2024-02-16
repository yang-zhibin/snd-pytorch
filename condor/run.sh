#!/bin/bash
# Login to WANDB
wandb login #WANDB_API_KEY

# Use user CERNBOX as EOS instance
export EOS_MGM_URL=root://eosuser.cern.ch
# stage-in
eos cp -p /eos/user/z/zhibin/graphNet/data/pt/pt_3class/train/* ./input_data/train/
eos cp -p /eos/user/z/zhibin/graphNet/data/pt/pt_3class/val/* ./input_data/val/
eos cp -p /eos/user/z/zhibin/graphNet/data/pt/pt_3class/test/* ./input_data/test/

nvidia-smi
source /afs/cern.ch/user/z/zhibin/work/graphNet/snd-pytorch/condor/env_conda.sh

#Issue of pytorch lightning 
export CUDA_VISIBLE_DEVICES=0,1

# Start training
python /afs/cern.ch/user/z/zhibin/work/graphNet/snd-pytorch/train_model.py /afs/cern.ch/user/z/zhibin/work/graphNet/snd-pytorch/Configs/evtReco_3class_full_train_norm.yaml