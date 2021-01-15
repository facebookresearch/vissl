Using Meters
===============================

VISSL supports PyTorch meters and implements some custom meters that like Mean Average Precision meter. Meters in VISSL support single target multiple outputs. This is especially useful and relvant during evaluation of self-supervised models where we want to calculate feature
quality of several layers of the model. See all the `VISSL custom meters here <https://github.com/facebookresearch/vissl/tree/master/vissl/meters>`_.

To use a certain meter, users need to simply set :code:`METERS.name=<my_meter_name>` and set the parameter values that meter requires.

Examples:

- Using Accuracy meter to compute Top-k accuracy for training and testing

.. code-block:: yaml

    METERS:
      name: accuracy_list_meter
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
