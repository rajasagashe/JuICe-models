# Juice Models
## Setup
Install conda. Then prepare environment:

    conda create -n {name} python=3.6 anaconda
    source activate {name}
    
    # install pytorch
    conda install pytorch torchvision cudatoolkit=8.0 -c pytorch
    
    # fairseq setup
    pip install --editable .

## Train/Evaluate LSTM
    CUDA_VISIBLE_DEVICES=0 python run.py  --model-dir {model directory} --dataset-dir {directory of downloaded dataset} --max-tokens 12000 --max-ctx-cells 3 --max-ctx-cell-tokens 75 --max-seq-len 250 -model lstm --code-key code_tokens_clean --num-merges 10000 --train-max 100000 

## Train/Evaluate Transformer
Using 3 gpus yields a bigger batch size.

    CUDA_VISIBLE_DEVICES=0,1,2 python run.py  --model-dir {model directory} --dataset-dir {directory of downloaded dataset}  --max-tokens 11000 --max-ctx-cells 3  -model transformer --code-key code_tokens_clean --num-merges 10000 --max-ctx-cell-tokens 50 --train-max 100000  --max-seq-len 250  --lr .0001 --warmup 1000

## Viewing Predictions
During evaluation (generate_jupyter.py) the first 100 dev/test model predictions will be logged jupyter notebook under the model directory. This streamlines performing error analysis. 


### Acknowledgments
This code is built on the [fairseq](https://github.com/pytorch/fairseq) library.

















