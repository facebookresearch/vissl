Using Meters
===============================

VISSL supports PyTorch meters and implements some custom meters like Mean Average Precision meter. Meters in VISSL support multi target and multiple outputs. This is especially useful and relevant during the evaluation of self-supervised models where we want to measure feature
quality of several layers of the model. See all the `VISSL custom meters here <https://github.com/facebookresearch/vissl/tree/main/vissl/meters>`_.

To use a certain meter, users need to simply set :code:`METERS.name=<my_meter_name>` and set the parameter values that the meter requires. Users can also use multiple meters by setting :code:`METERS.names=["my_meter_name_one", "my_meter_name_two"]`.

Examples:

- Using Accuracy meter to compute Top-k accuracy for training and testing

.. code-block:: yaml

    METERS:
      name: "accuracy_list_meter"
      accuracy_list_meter:
        num_meters: 1          # number of outputs model has. also auto inferred
        topk_values: [1, 5]    # for each meter, what topk are computed.


- Using Mean AP meter:

.. code-block:: yaml

    METERS:
      name: mean_ap_list_meter
      mean_ap_list_meter:
        num_classes: 9605   # openimages v6 dataset classes
        num_meters: 1

- Using Precision@k:

.. code-block:: yaml

    METERS:
      name: precision_at_k_list_meter
      precision_at_k_list_meter:
        num_meters: 1
        topk_values: [1]

- Using Recall@k:

.. code-block:: yaml

    METERS:
      name: recall_at_k_list_meter
      recall_at_k_list_meter:
        num_meters: 1
        topk_values: [1]

- Using Multiple Meters

.. code-block:: yaml

    METERS:
      names: [recall_at_k_list_meter, precision_at_k_list_meter]
      precision_at_k_list_meter:
        num_meters: 1
        topk_values: [1]
      recall_at_k_list_meter:
        num_meters: 1
        topk_values: [1]
