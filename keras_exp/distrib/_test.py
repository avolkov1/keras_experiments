'''
'''
from __future__ import print_function

import sys
from time import sleep
import traceback

import tensorflow as tf

from keras_exp.multigpu import get_available_gpus
from keras_exp.distrib import TFClusterManagerFacade  # JobType, DevType


__all__ = ('test',)


# =============================================================================
# TEST DISTRIBUTED COMPUTE
# =============================================================================
def calcm():
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c = tf.matmul(a, b)
    return c


def distrib_graph(wgdev_list):
    # Creates a graph.
    clist = []
    for d in wgdev_list:
        with tf.device(d):
            clist.append(calcm())

    sum_ = tf.add_n(clist)
    return sum_


def test(cluster_parser_spec):
    scpar = cluster_parser_spec
    # Setting config on ther server instantiation and then re-using this same
    # config for sesssions is very important. This functionality is wrapped
    # in TFClusterManagerFacade.
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False,  # True,
                            allow_soft_placement=True,
                            gpu_options=gpu_options)

    cmgr_facade = TFClusterManagerFacade(
        scpar.num_tasks_per_host, scpar.hostnames,
        scpar.num_parameter_servers, scpar.my_proc_id)

    #: :type cluster_spec: tf.train.ClusterSpec
    cluster_spec = cmgr_facade.get_cluster_spec()
    # job_type = cmgr_facade.myjobtype
    # task_id = cmgr_facade.mytask_id

    # cspec_dict = cluster_spec.as_dict()
    # print('CLUSTER_SPEC_DICT: {}\n\tJOB_TYPE: {}\n\tTASK_ID: {}'
    #       '\n\tSERVER TARGET: {}\n\tIS_CHIEF: {}'
    #       .format(  # DEBUG
    #           cspec_dict, job_type, task_id, server.target, is_chief))

    # TF 1.2.x RDMA: specify protocol='grpc+verbs' in server below.
    server = cmgr_facade.get_server(config)  # , protocol='grpc+verbs')

    # if job_type == JobType.ps:
    #     # JOIN PARAMETER SERVERS
    #     # server.join()
    #     cmgr_facade.join(server)

    # Otherwise assumed worker
    # if job_type == JobType.worker:

    is_chief = cmgr_facade.is_chief

    # Once the server is started everything but the chief worker can join
    # the server and wait to process/service graph computations. Chief in this
    # test function pushes the compute graph.
    if not is_chief:
        # JOIN WORKERS (PS also) EXCEPT FOR CHIEF
        # server.join()
        cmgr_facade.join(server)

    # ps_tasks = cluster_spec.num_tasks(JobType.ps)
    # ps_device = '/job:ps/cpu:0'
    # ps_job_name = pydev.DeviceSpec.from_string(ps_device).job
    # ps_tasks = len(cspec_dict[ps_job_name])
    # print('PS_JOB_NAME: {}\nPS_TASKS: {}'.format(ps_job_name, ps_tasks))

    # The ngpus per host needs to be done with MPI or somehow sync'd. Currently
    # assuming all hosts have the same number of GPUs.
    gdev_list = get_available_gpus()
    ngpus = len(gdev_list)

    #: :type mywgdev: tf.DeviceSpec
    wgdev_list = cmgr_facade.get_allworkers_devlist(ngpus)
    # print('\n\tCLUSTER_SPEC_DICT: {}\n\tWGDEV_LIST: {}\n'
    #       .format(cmgr_facade.clusterspec_dict,
    #               [dev.to_string() for dev in wgdev_list]))  # DEBUG

    compute_graph = distrib_graph(wgdev_list)
    # config = server.server_def.default_session_config
    # with tf.Session(server.target, config=config) as sess:
    with cmgr_facade.get_session(server) as sess:
        # if not is_chief:
        #     # server.join()
        #     cmgr_facade.join(server, sess)

        sleep(2)  # Have the chief wait just in case. Occasionally get errors.
        # Perhaps implement a READY queue just like DONE queues.

        # ps_device = tf.DeviceSpec(job=JobType.ps,
        #                           device_type=DevType.cpu,
        #                           device_index=0).to_string()
        # ps_device = '/job:ps/cpu:0'
        # print('PS_DEVICE: {}'.format(ps_device))  # DEBUG
        # TO USE REPLICA WITH tf.train.Supervisor DO NOT JOIN WORKERS ABOVE.
        # USING IT BELOW FOR PRINTING "Hello,..." IS NOT NECESSARY.
        with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
            hello_tf = tf.constant("Hello, distributed TensorFlow!")
            result = sess.run(hello_tf)
            print('RESULT:\n{}\n'.format(result))

        while True:
            try:
                c = calcm()
                result = sess.run(c)
                print('RESULT NOT DISTRIBUTED:\n{}\n'.format(result))

                result = sess.run(compute_graph)
                print('RESULT DISTRIBUTED:\n{}\n'.format(result))
                break
            except Exception as err:
                traceback.print_exc()
                print('INHIBITING ERROR: {}'.format(err),
                      file=sys.stderr)
                continue

        # cmgr_facade.stop_chief(server, sess=sess)  # this works too

    cmgr_facade.stop_chief(server)

