Handling invalid images in dataloader
========================================

How VISSL solves it
---------------------
Self-supervised approaches like SimCLR, SwAV, etc that perform some form of contrastive learning contrast the features or cluster of one image with the other.
During the dataloading time, or in the training dataset itself, it's possible that there are invalid images. By default, in VISSL, when the dataloader
encounters an invalid image, a gray image is returned instead. Using gray images for the purpose of contrastive learning can lead to inferior model accuracy
especially if there are a lot of invalid images.

To solve this issue, VISSL provides a custom *base* dataset class called :code:`QueueDataset` that maintains 2 queues in CPU memory. One queue is used to enqueue valid seen images from previous minibatches and the other queue is used to dequeue. The :code:`QueueDataset` is implemented such that the same minibatch will never have the duplicate images. If we can't dequeue a valid image, we return None from the dequeue.
In short, :code:`QueueDataset` enables using the previously used valid images from the training in the current minibatch in place of invalid images.

Enabling QueueDataset
------------------------

VISSL makes it convenient for users to use the code:`QueueDataset` with simple configuration settings. To use the code:`QueueDataset`, users
need to set :code:`DATA.TRAIN.ENABLE_QUEUE_DATASET=true` and :code:`DATA.TEST.ENABLE_QUEUE_DATASET=true`.

Tuning the queue size of QueueDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
VISSL exposes the queue settings to configuration file that users can tune. The configuration settings are:


.. code-block:: yaml

    DATA:
      TRAIN:
        ENABLE_QUEUE_DATASET: True
      TEST:
        ENABLE_QUEUE_DATASET: True


.. note::

    If users encounter CPU out-of-memory issue, they might want to reduce the queue size
