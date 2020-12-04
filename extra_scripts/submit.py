import submitit
import os
import datetime
import argparse
import typing
from pathlib import Path

from tools import run_distributed_engines


class CheckpointWrapper:

    def __init__(self):
        pass

    def __call__(self, overrides):
        run_distributed_engines.hydra_main(overrides)

    def __submitit_checkpoint__(self, *args: typing.Any, **kwargs: typing.Any) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Send jobs to slurm via '
                                                 'Submitit. Args passed here '
                                                 'will override those in '
                                                 'vissl config file.')
    parser.add_argument('--config_file', type=str,
                        default='pretrain/supervised/supervised_128gpu_vit_b16_imagenet',
                        help='vissl config file')
    parser.add_argument('--job_name', type=str, default='vision_transformer')
    parser.add_argument('--time', default=2160, type=int, help='job time ' \
                                                            'request, in minutes')
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--gpus_per_task', type=int, default=8)
    parser.add_argument('--cpus_per_task', type=int, default=80)
    parser.add_argument('--ntasks_per_node', type=int, default=1)
    parser.add_argument('--mem', type=str, default='512GB', help='gigabytes '
                                                                 'of GPU ram per node')
    parser.add_argument('--partition', type=str, default='learnfair', \
                                                  help='learnfair, dev, or priority')
    parser.add_argument('--comment', type=str, default=None, help='Needed for priority')
    parser.add_argument('--run_id', type=str, default='60012',
                        help='Needed for multi-node jobs.')
    parser.add_argument('--batchsize_per_replica', type=int, default=16)
    # Note that a new subdirectory will be created for each job. See the
    # format below.
    parser.add_argument('--checkpoint_root', type=str, default=None, help='If none '
                                                               'provided, '
                                                               'will default '
                                                               'to current '
                                                                'directory.')

    parser.add_argument('--checkpoint_stem', type=str, default=None, help='If '
                                                                       'none '
                                                                          'provided, '
                                                                          'will default '
                                                                          'to current date and time')

    args = parser.parse_args()

    # See submitit slurm param options at:
    # https://github.com/facebookincubator/submitit/blob/e37fcf219e7aac0914a73f2642ada7a6d6c091c4/submitit/slurm/slurm.py#L387
    slurm_params = {
        'job_name': args.job_name,
        'partition': args.partition,
        'time': args.time,
        'nodes': args.nodes,
        # 'gpus_per_node': args.gpus_per_node,
        'mem': args.mem,
        'gpus_per_task': args.gpus_per_task,
        'ntasks_per_node': args.ntasks_per_node,
        'cpus_per_task': args.cpus_per_task
    }
    if args.comment:
        slurm_params['comment'] = args.comment

    # Create checkpoint directory
    checkpoint_root = ''
    if args.checkpoint_root:
        checkpoint_root = args.checkpoint_root
    if args.checkpoint_stem:
        checkpoint_directory = args.checkpoint_stem
    else:
        now = datetime.datetime.now()
        date_time_folder = now.strftime("%Y-%m-%d") + '/' + now.strftime("%H-%M-%S")
        checkpoint_directory = f'{args.job_name}/{date_time_folder}'
    checkpoint_directory = os.path.join(checkpoint_root, checkpoint_directory)
    Path(checkpoint_directory).mkdir(parents=True, exist_ok=True)

    executor = submitit.SlurmExecutor(folder=checkpoint_directory)
    executor.update_parameters(**slurm_params)

    # Create override parameter dictionary of format key = hierarchy in .yaml
    # config file, value = value.
    override_dict = {
        'CHECKPOINT.DIR': checkpoint_directory,
        'DISTRIBUTED.NUM_NODES': slurm_params['nodes'],
        # 'DISTRIBUTED.NUM_PROC_PER_NODE': slurm_params['gpus_per_node'],
        'DISTRIBUTED.RUN_ID': args.run_id,
        'DATA.TRAIN.BATCHSIZE_PER_REPLICA': args.batchsize_per_replica
    }
    # Create list of overrides to be passed as args
    overrides = []
    for k, v in override_dict.items():
        overrides.append(f'config.{k}={v}')
    overrides.append('hydra.verbose=True')
    overrides.append(f'config={args.config_file}')

    training_callable = CheckpointWrapper()
    job = executor.submit(training_callable, overrides)
    # en = submitit.JobEnvironment()
    # print(f'Job ID: {job.job_id}\n', overrides)
    print('See log file for details')

    print(job.results())

    # if args.value == 'test_loss':
    #     fxn = plot_input_surface.main
    # elif args.value == 'hessian':
    #     fxn = plot_hessian_eigen.main
    #
    # jobs = executor.map_array(fxn, param_sets)  # just a list of jobs
    #
    # for job, param_set in zip(jobs, param_sets):
    #     print(f'Job ID: {job.job_id}\n', param_set)
    #
    # for job in jobs:
    #     print(job.results())


