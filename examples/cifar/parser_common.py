'''
'''
from argparse import (
    HelpFormatter, RawDescriptionHelpFormatter,
    ArgumentParser)  # , FileType)
# ArgumentDefaultsHelpFormatter, SUPPRESS
from textwrap import dedent

from keras_exp.distrib.cluster_mgrs.tfclusterdefs import ProtocolType


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


# class CustomFormatter(ArgumentDefaultsHelpFormatter,
#                       RawDescriptionHelpFormatter, SmartFormatterMixin):
class CustomFormatter(RawDescriptionHelpFormatter, SmartFormatterMixin):
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
        # default=SUPPRESS,
        help='S|Run on multiple-GPUs using all available GPUs on a system. If'
        '\nnot passed does not use multiple GPU. If passed uses all GPUs.'
        '\nOptionally specify a number to use that many GPUs. Another approach'
        '\nis to specify CUDA_VISIBLE_DEVICES=0,1,... when calling script and'
        '\nspecify --mgpu to use this specified device list.'
        '\nThis option is only supported with TensorFlow backend.')

    parser.add_argument(
        '--enqueue', action='store_true', default=False,
        help='S|Use StagingArea in multi-gpu model. Default: %(default)s\n')

    parser.add_argument(
        '--syncopt', action='store_true', default=False,
        help='S|Use gradient synchronization in Optimizer. '
        'Default: %(default)s')

    parser.add_argument(
        '--nccl', action='store_true', default=False,
        help='S|Use NCCL contrib lib. Default: %(default)s')

    parser.add_argument(
        '--epochs', type=int, default=200,
        help='S|Number of epochs to run training for.\n'
        '(Default: %(default)s)\n')

    # parser.add_argument('--rdma', action='store_true', default=False,
    #                     help='S|Use RDMA with Tensorflow. Requires that TF\n'
    #                          'was compiled with RDMA support.\n')

    parser.add_argument(
        '--network', nargs='?', type=str, const=None, default=None,
        help='S|Network domain to use. Ex. several nodes with hostnames:\n'
        '    node1, node2.\n'
        'These are hooked up to networks:\n'
        '    node1.cm.cluster has address w.x.y.z \n'
        '    node1.ib.cluster has address w.x.y.z \n'
        '    node2.cm.cluster has address w.x.y.z \n'
        '    node2.ib.cluster has address w.x.y.z \n'
        'Then set option --network=cm.cluster or --network=ib.cluster\n'
        'Otherwise don''t specify and some default network will be used.')

    parser.add_argument(
        '--rdma', action='store', nargs='?', type=str.lower,
        const=ProtocolType.verbs, default=None,
        choices=[ProtocolType.verbs, ProtocolType.gdr],
        help='S|Use RDMA with Tensorflow. Requires that TF \n'
        'was compiled with RDMA support. If TF and infrastructure supports\n'
        'GPUDirect RDMA can specify gdr. Default: verbs when set.')

    return parser


def remove_options(parser, options):
    # ref: https://stackoverflow.com/a/36863647/3457624
    for option in options:
        for action in parser._actions:
            if vars(action)['option_strings'][0] == option:
                parser._handle_conflict_resolve(None, [(option, action)])
                break

