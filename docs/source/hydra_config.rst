YAML Configuration system
=========================


VISSL uses `Hydra <https://github.com/facebookresearch/hydra>`_ for configuration management. The configuration files are simple YAML files.
Hydra provides flexible yet powerful configuration system.

- Users can create configs for only a specific component of their training (for example: using different datasets) and overwrite a master configuration setting for that specific component. This way, Hydra allows reusability of configs.
- Hydra also allows to modify the configuration values from command line and
- Hydra also offers an intuitive solution to adding new keys to a configuration.

The usage looks like:

.. code-block:: bash

    python <binary-name>.py config=<yaml_config path>/<yaml_config_file_name>
    
**All the parameters and settings VISSL supports**: you can see all the settings in `VISSL defaults.yaml file <https://github.com/facebookresearch/vissl/blob/master/vissl/config/defaults.yaml>`_.


Detecting new configuration directories in Hydra
------------------------------------------------------

VISSL provides configuration files `here <https://github.com/facebookresearch/vissl/tree/master/configs>`_ and uses the Hydra Plugin `VisslPlugin <https://github.com/facebookresearch/vissl/blob/master/hydra_plugins/vissl_plugin/vissl_plugin.py>`_
to automatically search for the :code:`configs` folder in VISSL.

If users want to create their own configuration directories and not use the :code:`configs` directory provided by VISSL, then users must
add their own Plugin following the :code:`VisslPlugin`.

.. note::

    For any new folder containing configuration files, Hydra requires creating a :code:`__init__.py` empty file. Hence, if users
    create a new configuration directory, they must create empty :code:`__init__.py` file.


How to use VISSL provided config files
----------------------------------------

For example, to train SwAV model on 8-nodes (32-gpu) with VISSL:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/swav/swav_8node_resnet

where :code:`swav_8node_resnet.yaml` is a master configuration file for SwAV training and exists at :code:`vissl/configs/config/pretrain/swav/swav_8node_resnet.yaml`.


How to add configuration files for new SSL approaches
-------------------------------------------------------

Let's say you have a new self-supervision approach that you implemented in VISSL and want to create config files for training. You can simply create a new folder and config file for your approach.

For example:

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=pretrain/my_new_approach/my_approach_config_file.yaml

In the above case, we are simply
creating the :code:`my_new_approach` folder under :code:`pretrain/` path and create a file :code:`my_approach_config_file.yaml` with the path `pretrain/my_new_approach/my_approach_config_file.yaml`


How to override a training component with config files
---------------------------------------------------------

To replace one training component with the other, for example, replacing the training datasets, one can achieve this by simply
creating a new yaml file for the dataset and use that during training.

For example:

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=pretrain/swav/swav_8node_resnet \
      +config/pretrain/swav/optimization=my_new_optimization \
      +config/pretrain/swav/my_new_dataset=my_new_dataset_file_name \

In the above case, we are overriding optimization and data settings for the SwAV training. For overriding, we simply
create the :code:`my_new_dataset` sub-folder under :code:`pretrain/swav` path and create a file :code:`my_new_dataset_file_name.yaml` with the path `pretrain/swav/my_new_dataset_file_name.yaml`


How to override single values in config files
-----------------------------------------------

If you want to override single value of an existing key in the config, you can achieve that with: :code:`my_key=my_new_value`

For example:

.. code-block:: bash

    python tools/run_distributed_engines.py \
        config=pretrain/swav/swav_8node_resnet \
        config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights_path.torch>


How to add new keys to the dictionary in config files
------------------------------------------------------

If you want to add single key to a dictionary in the config, you can achieve that with :code:`+my_new_key_name=my_value`. Note the use of :code:`+`.

For example:

.. code-block:: bash

    python tools/run_distributed_engines.py \
        config=pretrain/swav/swav_8node_resnet \
        +config.MY_NEW_KEY=MY_VALUE \
        +config.LOSS.simclr_info_nce_loss.MY_NEW_KEY=MY_VALUE
