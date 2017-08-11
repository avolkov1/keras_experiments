'''
'''
from argparse import (
    ArgumentDefaultsHelpFormatter, HelpFormatter, RawDescriptionHelpFormatter,
    ArgumentParser, SUPPRESS)  # , FileType)
from textwrap import dedent


__all__ = ('parser_def_mgpu', 'remove_options',)


class SmartFormatterMixin(HelpFormatter):
    # ref:
    # http://stackoverflow.com/questions/3853722/python-argparse-how-to-insert-newline-in-the-help-text
    # @IgnorePep8

    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('S|'):
            return text[2:].splitlines()
        return HelpFormatter._split_lines(self, text, width)


class CustomFormatter(ArgumentDefaultsHelpFormatter,
                      RawDescriptionHelpFormatter, SmartFormatterMixin):
    '''Convenience formatter_class for argparse help print out.'''


def parser_def_mgpu(desc):
    '''Define a prototyp parser with multi-gpu options and other common
    arguments.
    '''
    parser = ArgumentParser(description=dedent(desc),
                            formatter_class=CustomFormatter)

    # parser.add_argument(
    #     '--mgpu', action='store_true',
    #     help='S|Run on multiple-GPUs using all available GPUs on a system.')

    parser.add_argument(
        '--mgpu', action='store', nargs='?', type=int,
        const=-1,  # if mgpu is specified but value not provided then -1
        # if mgpu is not specified then defaults to 0 - single gpu
        # mgpu = 0 if getattr(args, 'mgpu', None) is None else args.mgpu
        default=SUPPRESS,
        help='S|Run on multiple-GPUs using all available GPUs on a system. If'
        '\nnot passed does not use multiple GPU. If passed uses all GPUs.'
        '\nOptionally specify a number to use that many GPUs. Another approach'
        '\nis to specify CUDA_VISIBLE_DEVICES=0,1,... when calling script and'
        '\nspecify --mgpu to use this specified device list.'
        '\nThis option is only supported with TensorFlow backend.')

    parser.add_argument('--enqueue', action='store_true', default=False,
                        help='S|Use StagingArea in multi-gpu model.\n')

    parser.add_argument('--syncopt', action='store_true', default=False,
                        help='S|Use gradient synchronization in Optimizer.\n')

    parser.add_argument('--nccl', action='store_true', default=False,
                        help='S|Use NCCL contrib lib.\n')

    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to run training for.')

    parser.add_argument('--rdma', action='store_true', default=False,
                        help='S|Use RDMA with Tensorflow. Requires that TF \n'
                             'was compiled with RDMA support.\n')

    return parser


def remove_options(parser, options):
    # ref: https://stackoverflow.com/a/36863647/3457624
    for option in options:
        for action in parser._actions:
            if vars(action)['option_strings'][0] == option:
                parser._handle_conflict_resolve(None, [(option, action)])
                break

