'''
'''

__all__ = ('JobType', 'DevType', 'ProtocolType',)


class JobType(object):
    worker = 'worker'
    ps = 'ps'  # parameter-server


class DevType(object):
    cpu = 'cpu'
    gpu = 'gpu'


class ProtocolType(object):
    '''
    :cvar grpc: 'grpc' default in Tensorflow
    :cvar verbs: 'verbs' # RDMA Infiniband usually
    :cvar gdr: 'gdr' GPUDirect RDMA
    '''
    grpc = 'grpc'  # default in Tensorflow
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/verbs/README.md
    verbs = 'verbs'  # RDMA Infiniband usually
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gdr/README.md
    gdr = 'gdr'  # GPUDirect RDMA

    @classmethod
    def get_server_protocol_str(cls, protocol_type):
        '''
        :param str protocol_type: Expects one of class variables belonging to
            ProtocolType but does not check.
        '''
        if not protocol_type:
            # blank string or None
            return None

        server_protocol_str = cls.grpc  # default
        if protocol_type != cls.grpc:
            server_protocol_str = '{}+{}'.format(cls.grpc, protocol_type)

        return server_protocol_str
