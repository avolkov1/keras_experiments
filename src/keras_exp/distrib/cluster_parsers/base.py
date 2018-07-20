'''
'''
from abc import ABCMeta, abstractproperty


__all__ = ()


ABC = ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3


# @six.add_metaclass(ABCMeta)
class ClusterParser(ABC):
    '''ClusterParser Abstract Base Class. Defines the interface expected
    of a cluster parser.
    '''

    @abstractproperty
    def num_tasks_per_host(self):
        '''List of integers. Length of list is number of hosts. Each list
        element specifies number of tasks on the host. A corresponding
        property hostnames is a list of hosts with each element specifying
        the host name. This list and hostnames list must be in the same order.
        '''

    @abstractproperty
    def hostnames(self):
        '''List of hosts with each element specifying the host name.'''

    @abstractproperty
    def num_parameter_servers(self):
        '''Number of parameter servers to create/use in the cluster.'''

    @abstractproperty
    def my_proc_id(self):
        '''Current process's id or rank.'''

    @abstractproperty
    def starting_port(self):
        '''Port to start from.'''
