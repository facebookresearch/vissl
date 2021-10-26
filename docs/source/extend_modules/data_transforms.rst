Add new Data Transforms
=======================

Adding new transforms and using them is quite easy in VISSL. Follow the steps below:

- **Step1**: Create your transform under :code:`vissl/data/ssl_transforms/my_new_transform.py`. The transform should follow the template:


.. code-block:: python


    @register_transform("MyNewTransform")
    class MyNewTransform(ClassyTransform):
        """
        add documentation for what your transform does
        """

        def __init__(self, param1, param2, ...):
            """
            Args:
                param1: add doctring
                param2: add doctring
                ...
            """
            self.param1 = param1
            self.param2 = param2
            # implement anything that the transform init should do
            ...

        # the input image should either be Image.Image PIL instance or torch.Tensor
        def __call__(self, image: {Image.Image or torch.Tensor}):
            # implement the transformation logic code.
            return img

        @classmethod
        def from_config(cls, config: Dict[str, Any]) -> "MyNewTransform":
            """
            Instantiates MyNewTransform from configuration.

            Args:
                config (Dict): arguments for for the transform

            Returns:
                MyNewTransform instance.
            """
            param1 = config.param1
            param2 = config.param2
            ....
            return cls(param1=param1, param2=param2, ...)

- **Step2**: Use your transform in the config file by editing the :code:`DATA.TRAIN.TRANSFORMS` value:

.. code-block:: yaml

    DATA:
      TRANSFORMS:
        ...
        ...
        - name: MyNewTransform
          param1: value1
          param2: value2
          ....
        ....
