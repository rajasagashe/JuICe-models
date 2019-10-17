import argparse
import json
import sys

from fairseq.data import Dictionary
from fairseq.subword_nmt import learn_bpe, apply_bpe

def jloadl(filename, lines=-1):
    with open(filename) as fobj:
        for i, line in (enumerate(fobj)):
            if i == lines:
                break
            yield json.loads(line)


def get_input(js, code_key, max_ctx_cell_tokens, max_ctx_cells, use_comment):
    '''Concatenates all the context cells above into 1 large input sequence.'''
    seq2seq = []
    nl_first = True

    if max_ctx_cells > 1:
        # if 'boilerplate_code_tokens' in js:
        #     seq2seq.append('CODE')
        #     seq2seq.extend(js['boilerplate_code_tokens'])
        ##     boilerplate is treated as a context cell so we decrement
        ##     to compensate for logic below.
        #    max_ctx_cells -= 1

        for rec in js['context'][:max_ctx_cells]:
            # separator token
            seq2seq.append(rec['cell_type'].upper())

            if rec['cell_type'] == 'code':
                code_toks = rec[code_key]


                if code_toks:
                    code_toks = code_toks[:max_ctx_cell_tokens]
                    seq2seq.extend(code_toks)
            elif rec['cell_type'] == 'markdown':
                nl_toks = rec['nl']
                if nl_first:
                    # the target nl shouldn't be truncated since its needed to
                    # generate the code.
                    nl_toks = nl_toks[-max_ctx_cell_tokens:]
                else:
                    nl_toks = nl_toks[-max_ctx_cell_tokens:]
                nl_first = False
                # this would truncate the nl, or full code tokens
                seq2seq.extend(nl_toks)
    else:
        # for 1 ctx len we make sure that nl is used since in dev and test, the nl
        # may be more than 1 above. this occurs for a small fraction of cases. remove
        # this later since not that useful.

        seq2seq.append('MARKDOWN')
        seq2seq.extend(js['nl'][-max_ctx_cell_tokens:])

    # seq2seq += ['IMPORT'] + js['imports'][:max_ctx_cell_tokens]
    if use_comment:
        # issues: comment code will get added. this could make comments very long
        # so thats why safe to truncate
        seq2seq = ['COMMENT'] + js['comments'][:max_ctx_cell_tokens] + seq2seq

    return seq2seq


def get_bpe(lst, num_merges):
    '''Learn bpe'''
    codes = learn_bpe.learn_bpe(lst, num_merges, 0) # last is min freq
    assert len(codes) == num_merges

    # don't need to write em out
    # with open(codes_file, 'w') as fd:
    #     json.dump(codes, fd, indent=4)
    bpe = apply_bpe.BPE(codes)
    return bpe

def load_dataset(args, split):
    '''Load it and concatenate all context cells as input sequence.'''

    # we dont want to truncate dev and test in rare case train_max is less than size of those
    max_num = args.train_max if split == 'train' else sys.maxsize
    nl = []
    code = []
    urls = []
    for i, js in enumerate(jloadl(args.dataset_dir + f'/{split}.jsonl', max_num)):
        nl.append(get_input(js, code_key=args.code_key,
                            max_ctx_cell_tokens=args.max_ctx_cell_tokens,
                            max_ctx_cells=args.max_ctx_cells,
                            use_comment=args.use_comment))
        code.append(js[args.code_key])
        urls.append(js['metadata']['url'] if 'url' in js['metadata'] else js['metadata']['nb_orig_url'])

    return nl, code, urls

def bpe_segment(bpe, dataset):
    return [bpe.segment_tokens(toks) for toks in dataset]

def write_dictionary(model_dir, lst):
    '''Write out dictionary in fair seq format.'''
    joined_dict = Dictionary()
    for toks in lst:
        for t in toks:
            joined_dict.add_symbol(t)

    print('| dictionary: {} types'.format(len(joined_dict)))

    with open(model_dir + '/dict.txt', 'w') as fd:
        joined_dict.save(fd)

def dump_dataset(model_dir, nl, code, urls, split):
    '''Write out a split in which each record is line delimited and
    each token is space separated.'''
    nl_file = open(model_dir + f'/{split}.nl', 'w')
    code_file = open(model_dir + f'/{split}.code', 'w')
    url_file = open(model_dir + f'/{split}.url', 'w')

    max_seq_len = 255
    num_skipped = 0
    for n, c, u in zip(nl, code, urls):
        if len(n) > max_seq_len:
            num_skipped += 1
        #     continue
        if len(c) > max_seq_len:
            num_skipped += 1
        #     continue
        nl_file.write(' '.join(n) + '\n')
        code_file.write(' '.join(c) + '\n')
        url_file.write(u + '\n')

    print(f'| Dumping {split}...')
    print('| avg input len {:.2f}'.format(sum([len(t) for t in nl]) / len(nl)))
    print('| avg output len {:.2f}'.format(sum([len(t) for t in code])/ len(code)))
    print(f'| Num longer than max_seq_len {split}: {num_skipped} as a fraction: {num_skipped/len(nl)}')

def preprocess(args):
    print('==================== Preprocess')
    # print('| Loading train...')
    train_nl, train_code, urls = load_dataset(args, 'train')
    print('| Learning Bpe...')
    bpe = get_bpe([w for lst in (train_nl + train_code) for w in lst], args.num_merges)

    train_nl = bpe_segment(bpe, train_nl)
    train_code = bpe_segment(bpe, train_code)

    write_dictionary(args.model_dir, train_nl + train_code)

    dump_dataset(args.model_dir, train_nl, train_code, urls, 'train')


    valid_nl, valid_code, urls = load_dataset(args, 'dev')
    valid_nl = bpe_segment(bpe, valid_nl)
    valid_code = bpe_segment(bpe, valid_code)
    dump_dataset(args.model_dir, valid_nl, valid_code, urls, 'valid')


    test_nl, test_code, urls = load_dataset(args, 'test')
    test_nl = bpe_segment(bpe, test_nl)
    test_code = bpe_segment(bpe, test_code)
    dump_dataset(args.model_dir, test_nl, test_code, urls, 'test')

    print(f'| Preprocessed dataset under {args.model_dir}')

def cli_main():
    parser = argparse.ArgumentParser(description='run.ipy')
    parser.add_argument('--model-dir', type=str, default='',
                        help='Models directory.')
    parser.add_argument('--dataset-dir', type=str, default='',
                        help='train data directory.')

    parser.add_argument('--max-ctx-cells', default=1, type=int, required=True,
                        help='Num cells to use above')
    parser.add_argument('--max-ctx-cell-tokens', default=50, type=int, required=True,
                        help='Num cells to use above')
    parser.add_argument('--use-comment', action='store_true',
                        help='')

    parser.add_argument('--num-merges', type=int, required=True,
                        help='train data directory.')
    parser.add_argument('--train-max', type=int, required=True,
                        help='train data directory.')
    parser.add_argument('--code-key', type=str, required=True)

    # parser.add_argument('--max-seq-len', type=int, required=True,
    #                     help='train data directory.')
    args = parser.parse_args()
    preprocess(args)

