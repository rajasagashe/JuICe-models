''' Used to put preds and target into a notebook. Why notebook? so any annotations
can be entered in the cell and stored.
'''

import json
import random
from pathlib import Path

random.seed(1)

def new_markdown_cell(source=None):
    return {
        'cell_type': 'markdown',
        'source': source if source else '',
        'metadata': {}
    }
def new_code_cell(source=None):
    return {
        'cell_type': 'code',
        'source': source if source else '',
        'metadata': {},
        'outputs': [],
        'execution_count': 0
    }
def new_notebook(cells):
    return  {
        'cells': cells,
        'metadata': {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.6.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def get_filtered_preds(preds, function):
    cells = []
    # for p in preds:
    for p in filter(function, preds):
        # the logging format will be markdown with url, then src and tgt in a code block
        # followed by an empty code cell for annotation
        cells.append(new_markdown_cell(f"### {p['id']}"))
        # cells.append(new_markdown_cell(p['url']))
        cells.append(new_markdown_cell('Source: ' + p['src_str']))

        # these special tokens will be there in full code but not api sequence
        # so always replace them for readability
        cells.append(new_code_cell('Target    : ' + p['tgt_str'].replace('NEWLINE', '\n').replace('INDENT', '    ')))
        # print(p['tgt_str'].replace('NEWLINE', '\n'))
        cells.append(new_code_cell('Hypothesis: ' + p['hypo_str'].replace('NEWLINE', '\n').replace('INDENT', '    ')))
        cells.append(new_code_cell("whats correct: "))
        cells.append(new_code_cell("whats incorrect: "))
        cells.append(new_code_cell("whats needed: "))

    return cells


def log_preds_to_notebook(preds, outdir, randomize=False, trunc=100, func=None, code_key='code_tokens_clean'):
    ''' Writes out preds into notebooks, also a notebook with em corrects written otu.
    :param preds:
    :param outfile:
    :param randomize: the dev/test shuffled before hand so no need to
    :return:
    '''
    if randomize:
        random.shuffle(preds)

    preds_trunc = preds[:trunc]
    Path(outdir).mkdir(parents=True, exist_ok=True)

    cells = get_filtered_preds(preds_trunc, lambda x: True)
    json.dump(new_notebook(cells), open(outdir + '/preds.ipynb', 'w'), indent=4)

    cells = get_filtered_preds(preds, lambda x: x['hypo_str'] == x['tgt_str'])
    json.dump(new_notebook(cells), open(outdir + '/preds_em_correct.ipynb', 'w'), indent=4)

    # cells = get_filtered_preds(preds, lambda x: any([t in x['tgt_str'] for t in x['hypo_str'].split()]))
    # json.dump(new_notebook(cells), open(outdir + '/preds_partial_correct.ipynb', 'w'), indent=4)
