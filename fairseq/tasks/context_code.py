import os

from fairseq.data import LanguagePairDataset
from fairseq.tasks import FairseqTask, register_task


@register_task('context_code')
class ContextCodeTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--max-seq-len', type=int, required=True,
                            help='Drop the example if either nl/code longer than this.')

        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        # parser.add_argument('--max-source-positions', default=300, type=int, metavar='N',
        #                     help='max number of tokens in the source sequence')
        # parser.add_argument('--max-target-positions', default=300, type=int, metavar='N',
        #                     help='max number of tokens in the target sequence')

    @classmethod
    def setup_task(cls, args, **kwargs):
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just load the Dictionaries.

        # train command has save_dir while evaluate has path to the model
        model_dir = args.save_dir if hasattr(args, 'save_dir') else os.path.dirname(args.path)
        dict_file = model_dir + '/dict.txt'

        joined_dict = cls.load_dictionary(dict_file)
        return ContextCodeTask(args, joined_dict)


    def __init__(self, args, shared_dict):
        super().__init__(args)
        self.shared_dict = shared_dict

    def load_dataset(self, split, **kwargs):
        """This is basically the same as the translation task, except we also use
        the urls."""

        # todo load urls then just call translation task

        nl = open(self.args.data + f'/{split}.nl').read().splitlines()
        code = open(self.args.data + f'/{split}.code').read().splitlines()
        urls = open(self.args.data + f'/{split}.url').read().splitlines()

        self.urls = []
        nl_toks = []
        code_toks = []
        for n, c, u in zip(nl, code, urls):

            self.urls.append(u)

            nl_toks.append(self.shared_dict.encode_line(
                n, add_if_not_exist=False,
            ).long())

            code_toks.append(self.shared_dict.encode_line(
                c, add_if_not_exist=False,
            ).long())

        assert len(nl_toks) == len(code_toks)
        print('| {} {} {} examples'.format(self.args.data, split, len(nl_toks)))

        # We reuse LanguagePairDataset since classification can be modeled as a
        # sequence-to-sequence task where the target sequence has length 1.
        self.datasets[split] = LanguagePairDataset(
            src=nl_toks,
            src_sizes=[t.numel() for t in nl_toks],
            src_dict=self.shared_dict,
            tgt=code_toks,
            tgt_sizes=[t.numel() for t in code_toks],  # targets have length 1
            tgt_dict=self.shared_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_seq_len,
            max_target_positions=self.args.max_seq_len,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_seq_len, self.args.max_seq_len)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.shared_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.shared_dict
