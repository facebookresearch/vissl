Add new Data Collators
=======================

VISSL allows implementing new data collators easily. Follow the steps below:

- **Step1**: Create the new data collator :code:`my_new_collator.py` under :code:`vissl/data/collators/my_new_collator.py` following the template.

.. code-block:: python

    import torch
    from vissl.data.collators import register_collator

    @register_collator("my_new_collator")
    def my_new_collator(batch, param1 (Optional), ...):
        """
        add documentation on what new collator does

        Input:
            add documentation on what input type should the collator expect. i.e
            what should the `batch' look like.

        Output:
            add documentation on what the collator returns i.e. what does the
            collated data `output_batch' look like.
        """
        # implement the collator
        ...
        ...

        output_batch = {
            "data": ... ,
            "label": ... ,
            "data_valid": ... ,
            "data_idx": ... ,
        }
        return output_batch

- **Step2**: Use your new collator via the configuration files

.. code-block:: yaml

    DATA:
      TRAIN:
        COLLATE_FUNCTION: my_new_collator
        COLLATE_FUNCTION_PARAMS: {...}  # optional, specify params if collator requires any
