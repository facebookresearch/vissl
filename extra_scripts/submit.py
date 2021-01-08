import submitit
import os
import datetime
import argparse
import typing
from pathlib import Path
import itertools

from tools import run_distributed_engines


class CheckpointWrapper:

    def __init__(self):
        pass

    def __call__(self, overrides):
        run_distributed_engines.hydra_main(overrides)

    def checkpoint(self, *args: typing.Any, **kwargs: typing.Any) -> submitit.helpers.DelayedSubmission:
        training_callable = CheckpointWrapper()
        return submitit.helpers.DelayedSubmission(training_callable, *args, **kwargs)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


if __name__ == '__main__':
    # Providing multiple parameter values separated by commas (e.g.
    # --batchsize_per_replica 48,96 --OPTIMIZER.weight_decay 0.3,0.1,
    # 0.05) will submit a job array with a job for each parameter combination
    parser = argparse.ArgumentParser(description='Send jobs to slurm via '
                                                 'Submitit. Args passed here '
                                                 'will override those in '
                                                 'vissl config file.')
    parser.add_argument('--config_file', type=str,
                        default='pretrain/supervised/supervised_128gpu_vit_b16_imagenet',
                        help='vissl config file')
    parser.add_argument('--job_name', type=str, default='vision_transformer')
    parser.add_argument('--time', default=2880, type=int, help='job time ' \
                                                            'request, in minutes')
    parser.add_argument('--nodes', type=int, default=4)
    parser.add_argument('--gpus_per_task', type=int, default=8)
    parser.add_argument('--cpus_per_task', type=int, default=80)
    parser.add_argument('--ntasks_per_node', type=int, default=1)
    parser.add_argument('--mem', type=str, default='479GB', help='gigabytes '
                                                                 'of GPU ram per node')
    parser.add_argument('--partition', type=str, default='learnfair', \
                                                  help='learnfair, dev, or priority')
    parser.add_argument('--max_num_timeout', type=int, default=12,
                        help='Maximum number of resubmissions on '
                             'timeout/preemption')
    parser.add_argument('--comment', type=str, default=None, help='Needed for priority')
    parser.add_argument('--constraint', type=str, default=None, help='e.g. volta32gb')
    parser.add_argument('--run_id', type=str, default='60012',
                        help='Needed for multi-node jobs.')
    parser.add_argument('--batchsize_per_replica', type=int)
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

    args, unparsed_overrides = parser.parse_known_args()
    additional_overrides = {}

    # Iterate over every unparsed parameter-argument pair and add them to a
    # dictionary that will be added to the config overrides
    for arg_ind in range(0, len(unparsed_overrides), 2):
        arg = unparsed_overrides[arg_ind]
        if arg.startswith(('--')):
            curr_param = arg.split('--')[1]
        elif arg.startswith(('-')):
            curr_param = arg.split('-')[1]
        elif arg.startswith('+'):
            curr_param = arg
        additional_overrides[curr_param] = unparsed_overrides[arg_ind + 1]

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
    if args.constraint:
        slurm_params['constraint'] = args.constraint

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

    executor = submitit.SlurmExecutor(folder=checkpoint_directory,
                                      max_num_timeout=args.max_num_timeout)
    executor.update_parameters(**slurm_params)

    # Create override parameter dictionary of format key = hierarchy in .yaml
    # config file, value = value.
    override_dict = {
        'CHECKPOINT.DIR': checkpoint_directory,
        'DISTRIBUTED.NUM_NODES': str(slurm_params['nodes']),
        # 'DISTRIBUTED.NUM_PROC_PER_NODE': slurm_params['gpus_per_node'],
        'DISTRIBUTED.RUN_ID': args.run_id
    }
    if args.batchsize_per_replica:
        override_dict['DATA.TRAIN.BATCHSIZE_PER_REPLICA'] = args.batchsize_per_replica
    override_dict.update(additional_overrides)
    override_dict['config_file'] = args.config_file
    # Iterate through each argument and check if multiple values have been
    # passed. If so, split on comma. If not, put in list for convenience
    # when using itertools.product.
    for k, v in override_dict.items():

        if ',' in v:
            override_dict[k] = v.split(',')
        else:
            override_dict[k] = [v]
    jobs = []
    job_params_to_print = []
    checkpoint_directories = []
    with executor.batch():
        n_param_combinations = len(list(product_dict(**override_dict)))
        # Iterate through sets of parameters
        for i, param_dict in enumerate(product_dict(**override_dict)):
            # Create list of overrides to be passed as args
            overrides = []
            for k, v in param_dict.items():
                if k.startswith('+'):
                    overrides.append(f'{k}={v}')
                elif k == 'config_file':
                    overrides.append(f'config={v}')
                else:
                    # Create subdirectory for each job in array
                    if k == 'CHECKPOINT.DIR':
                        if n_param_combinations > 1:
                            v = f'{v}/{i}'
                        checkpoint_directories.append(v)
                    overrides.append(f'config.{k}={v}')
            overrides.append('hydra.verbose=True')
            training_callable = CheckpointWrapper()
            job = executor.submit(training_callable, overrides)
            jobs.append(job)
            job_params_to_print.append('\n'.join(overrides))

    # en = submitit.JobEnvironment()
    # print(f'Job ID: {job.job_id}\n', overrides)\
    for i, j in enumerate(zip(jobs, checkpoint_directories, job_params_to_print)):
        slurm_job = j[0]
        checkpoint_directory = j[1]
        job_params = j[2]
        print(f'\nJob {i}\nSlurm id: {slurm_job.job_id}\nLog file: '
              f'{checkpoint_directory}/log.txt')
        print(f'Overrides:\n{job_params}')

    outputs = [job.results() for job in jobs]

    for output in outputs:
        print(output)

    # print(f'See log file for details: {checkpoint_directory}/log.txt')

    # print(job.results())

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


