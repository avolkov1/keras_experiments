# Taken from: https://github.com/jhollowayj/tensorflow_slurm_manager
# ref:
# https://github.com/jhollowayj/tensorflow_slurm_manager/blob/master/slurm_manager.py
# @IgnorePep8
'''
'''
from __future__ import print_function

import os
import re

import hostlist

from ._distrib import ClusterParser


__all__ = ('SlurmClusterParser',)


# It may be useful to know that slurm_nodeid tells you which node you are one
# (in case there is more than one task on any given node...)
# Perhaps you could better assign parameter servers be distributed across all
# nodes before doubleing up on one.
class SlurmClusterParser(ClusterParser):
    '''
        :param num_param_servers: Default -1 meaning one parameter server per
            node. The remaining processes on the node are workers. The
            num_parameter_servers be less than or equal to the number of
            individual physical nodes

        :param starting_port: Starting port for setting up jobs. Default: 2222
            TODO: Maybe use SLURM_STEP_RESV_PORTS environment if available.
                https://stackoverflow.com/a/36803148/3457624
    '''
    def __init__(self, num_param_servers=-1, starting_port=2222):
        num_workers = None
        # Check Environment for all needed SLURM variables
        # SLURM_NODELIST for backwards compatability if needed.
        assert 'SLURM_JOB_NODELIST' in os.environ
        assert 'SLURM_TASKS_PER_NODE' in os.environ
        assert 'SLURM_PROCID' in os.environ
        assert 'SLURM_NPROCS' in os.environ
        assert 'SLURM_NNODES' in os.environ

        # Grab SLURM variables
        # expands 'NAME1(x2),NAME2' -> 'NAME1,NAME1,NAME2'
        self._hostnames = hostlist.expand_hostlist(
            os.environ['SLURM_JOB_NODELIST'])
        # expands '1,2(x2)' -> '1,2,2'
        self._num_tasks_per_host = self._parse_slurm_tasks_per_node(
            os.environ['SLURM_TASKS_PER_NODE'])
        # index into hostnames/num_tasks_per_host lists
        self._my_proc_id = int(os.environ['SLURM_PROCID'])
        self.num_processes = int(os.environ['SLURM_NPROCS'])
        self.nnodes = int(os.environ['SLURM_NNODES'])

        # Sanity check that everything has been parsed correctly
        nhosts = len(self.hostnames)
        assert nhosts == len(self.num_tasks_per_host)
        assert nhosts == self.nnodes
        assert self.num_processes == sum(self.num_tasks_per_host)

        # Numbber of PS/Workers
        # Note: I'm making the assumption that having more than one PS/node
        #       doesn't add any benefit. It makes code simpler in
        #       self.build_cluster_spec()
        self._num_parameter_servers = min(num_param_servers, nhosts) \
            if num_param_servers > 0 else nhosts

        if num_workers is None:
            # Currently I'm not using num_workers'
            # TODO: What happens to num_workers once I allocate less PS than
            #     they requested?
            # default to all other nodes doing something
            self.num_workers = self.num_processes - self.num_parameter_servers

        # Default port to use
        self.starting_port = starting_port  # use user specified port

    def _parse_slurm_tasks_per_node(self, num_tasks_per_nodes):
        '''
        SLURM_TASKS_PER_NODE Comes in compressed, so we need to uncompress it:
          e.g: if slurm gave us the following setup:
                   Host 1: 1 process
                   Host 2: 3 processes
                   Host 3: 3 processes
                   Host 4: 4 processes
        Then the environment variable SLURM_TASKS_PER_NODE = '1,3(x2),4'
        But we need it to become this => [1, 3, 3, 4]
        '''
        final_list = []
        num_tasks_per_nodes = num_tasks_per_nodes.split(',')

        for node in num_tasks_per_nodes:
            if 'x' in node:  # "n(xN)"; n=tasks, N=repeats
                n_tasks, n_nodes = [int(n) for n in re.findall('\d+', node)]
                final_list += [n_tasks] * n_nodes
            else:
                final_list.append(int(node))

        return final_list

    @property
    def num_tasks_per_host(self):
        '''List of integers with each element specifying number of tasks on a
        host. This list and hostnames list must be in the same order.'''
        return self._num_tasks_per_host

    @property
    def hostnames(self):
        '''List of hosts with each element specifying the host name.'''
        return self._hostnames

    @property
    def num_parameter_servers(self):
        '''Number of parameter servers to create/use in the cluster.'''
        return self._num_parameter_servers

    @property
    def my_proc_id(self):
        '''Current process's id or rank.'''
        return self._my_proc_id


if __name__ == '__main__':
    # run test via: python -m keras_exp.distrib.slurm
    from ._test import test
    scpar = SlurmClusterParser()
    test(scpar)
    # sys.exit(0)

