''' Use this file to preprocess the dataset, and train and evaluate LSTMs and Transformers
on the dataset.
'''

import argparse
import os
import shutil
import sys
from pathlib import Path

import generate_jupyter
import preprocess_jupyter
import train
from scripts import average_checkpoints


def train_lstm(models_dir, shared, max_tokens, max_seq_len):
    args = f"""{models_dir}

  -a lstm --optimizer adam --lr 0.001
  --encoder-bidirectional --encoder-layers 2 --decoder-layers 2
  --dropout 0.5
  --adam-betas (0.9,0.98) --clip-norm 5
  --lr-shrink 0.5
  --lr-scheduler reduce_lr_on_plateau
  --max-tokens {max_tokens}
  --save-dir  {models_dir}
  --no-progress-bar  --max-epoch 40 --keep-last-epochs 3

  {shared}

   --max-seq-len {max_seq_len}
   --skip-invalid-size-inputs-valid-test
  """

    sys.argv[1:] = args.split()
    # print(sys.argv)
    train.cli_main()

def train_transformer(models_dir, shared, max_tokens, max_seq_len, lr, warmup, dropout=0.3):

    args = f"""{models_dir}
  -a transformer_iwslt_de_en --optimizer adam --lr {lr}
  --label-smoothing 0.1 --dropout {dropout}
  --min-lr 1e-09 --lr-scheduler inverse_sqrt --weight-decay 0.0001
  --criterion label_smoothed_cross_entropy
  --warmup-updates {warmup} --warmup-init-lr 1e-07

  --tensorboard-logdir {models_dir +'/tensorboard'}

  --adam-betas (0.9,0.98)
  --share-all-embeddings
  --max-tokens {max_tokens}
  --save-dir  {models_dir}
  --max-epoch 300
  --no-progress-bar
  --keep-last-epochs 11

   --max-seq-len {max_seq_len}
   --skip-invalid-size-inputs-valid-test
  {shared}
  """
    # # if we're not averaging last checkpoints dont write them all out to save memory
    # if not checkpoint_avg:
    #     args += ' --no-epoch-checkpoints '

    sys.argv[1:] = args.split()
    # print(sys.argv)
    train.cli_main()


def run_generate(models_dir, shared, model_name='checkpoint_best.pt', subset='valid',
                 max_seq_len=sys.maxsize):
    args = f'''{models_dir}
     --path {models_dir}/{model_name}
     --max-tokens 8000
     --batch-size 128 --beam 5 --remove-bpe --gen-subset {subset}
     {shared}
     --max-seq-len {max_seq_len}
     --quiet
     '''

    sys.argv[1:] = args.split()
    # print(sys.argv)
    generate_jupyter.cli_main()

def checkpoint_average(models_dir, model_name='checkpoint_avg.pt'):
    model_name = 'checkpoint_avg.pt'
    sys.argv[1:] = f'''
        --inputs {models_dir}
        --num-epoch-checkpoints 10
        --output {models_dir}/{model_name}
        '''.split()
    average_checkpoints.main()

def preprocess(args, model_dir_name):
    args = f'''
     --model-dir {model_dir_name}
     --dataset-dir {args.dataset_dir}
     --max-ctx-cells {args.max_ctx_cells}
     --max-ctx-cell-tokens {args.max_ctx_cell_tokens}
     --num-merges {args.num_merges}
     --train-max {args.train_max}
     --code-key {args.code_key}
     --use-comment
     '''
    sys.argv[1:] = args.split()
    preprocess_jupyter.cli_main()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='run.ipy')

    # args for all
    parser.add_argument('--model-dir', type=str, default='',
                        help='Preprocessed dataset, dictionary, and models will be stored'
                             'in a new directory(created by this script and named after arguments)'
                             ' under this directory.')
    parser.add_argument('--dataset-dir', type=str, default='',
                        help='Location of train/dev/test data.')

    # preprocess args
    parser.add_argument('--max-ctx-cells', default=1, type=int, required=True,
                        help='Num cells to use above')
    parser.add_argument('--max-ctx-cell-tokens', default=50, type=int, required=True,
                        help='Num cells to use above')
    parser.add_argument('--code-key', type=str, required=True)
    parser.add_argument('--num-merges', type=int, default=1000, required=True,
                        help='train data directory.')
    parser.add_argument('--train-max', type=int, default=30000, required=True,
                        help='train data directory.')
    parser.add_argument('--max-seq-len', type=int, required=True,
                        help='train data directory.')

    # train args
    parser.add_argument('-model', type=str, default='',
                        help='')
    parser.add_argument('--eval', type=str,
                        help='')
    parser.add_argument('--human-subset', action='store_true',
                        help='')
    parser.add_argument('--checkpoint-avg', action='store_true',
                        help='')
    parser.add_argument('--max-tokens', type=int, default=12000, required=True,
                        help='train data directory.')
    parser.add_argument('--lr', type=float, default=.0005,
                        help='')
    parser.add_argument('--warmup', type=int, default=4000,
                        help='')

    opts = parser.parse_args()


    shared = f'--task context_code'

    # Create the model directory name based on the arguments.
    model_dir_name = opts.model_dir + f'/{opts.model}-train{opts.train_max}-merges{opts.num_merges}-ctx{opts.max_ctx_cells}-key{opts.code_key[:8]}'
    if opts.lr != .0005:
        model_dir_name += f'-lr{opts.lr}'
    if opts.warmup != 4000:
        model_dir_name += f'-warmup{opts.warmup}'
    # if opts.dropout != 0.3:
    #     model_dir_name += f'-dropout{opts.dropout}'



    if not opts.eval:
        # First check if its ok to delete this directory if it exists. Fairseq
        # supports easy recovery from checkpoints so that would just be another
        # prompt here.
        if os.path.exists(model_dir_name):
            choice = input('Directory already exists, delete [Y/n]? ')
            if choice == '':
                print('Deleting model directory...')
                assert (model_dir_name.startswith('/scratch/fairseq/jupyter/models') or
                model_dir_name.startswith('/exp/fairseq/jupyter/models'))
                shutil.rmtree(model_dir_name, ignore_errors=True)

            else:
                exit('Stopping...')

        Path(model_dir_name).mkdir(parents=True, exist_ok=True)

        preprocess(opts, model_dir_name)

        print('================= Train')
        if opts.model == 'transformer':
            train_transformer(model_dir_name, shared, opts.max_tokens, opts.max_seq_len, opts.lr, opts.warmup)
        elif opts.model == 'lstm':
            train_lstm(model_dir_name, shared, opts.max_tokens, opts.max_seq_len)

    model_name='checkpoint_best.pt'
    # if opts.checkpoint_avg:
    #     model_name = 'checkpoint_avg.pt'
    #     checkpoint_average(model_dir_name, model_name)
    #     # python scripts/average_checkpoints.py --inputs checkpoints/transformer  --num-epoch-checkpoints 10 --output checkpoints/transformer/model.pt

    print('================= Generate')
    split = opts.eval if opts.eval else 'valid'
    run_generate(model_dir_name, shared, model_name, subset=split)


