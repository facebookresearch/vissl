# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
import subprocess
from collections import OrderedDict
from typing import List

import numpy as np
import scipy.io
import torch
import torchvision.transforms.functional as TF
from iopath.common.file_io import g_pathmgr
from PIL import Image, ImageFile
from torch.nn import functional as F
from torchvision import transforms
from vissl.utils.instance_retrieval_utils.evaluate import (
    compute_map,
    score_ap_from_ranks_1,
)
from vissl.utils.io import load_file


def is_oxford_paris_dataset(dataset_name: str):
    """
    Computes whether the specified dataseet name is a revisited version of
    the oxford and paris datasets. simply looks for pattern "roxford5k"
    and "rparis6k" in specified dataset_name.
    """
    return dataset_name in ["Oxford", "Paris"]


def is_revisited_dataset(dataset_name: str):
    """
    Computes whether the specified dataseet name is a revisited version of
    the oxford and paris datasets. simply looks for pattern "roxford5k"
    and "rparis6k" in specified dataset_name.
    """
    return dataset_name in ["roxford5k", "rparis6k"]


def is_instre_dataset(dataset_name: str):
    """
    Returns True if the dataset name is "instre". Helper function used in code
    at several places.
    """
    return dataset_name == "instre"


def is_whiten_dataset(dataset_name: str):
    """
    Returns if the dataset specified has name "whitening". User can use any
    dataset they want for whitening.
    """
    return dataset_name == "whitening"


def is_copdays_dataset(dataset_name: str):
    """
    Is the dataset copydays.
    """
    return dataset_name in ["copydays"]


# pooling + whitening
# Credits: Matthijs Douze
def add_bias_channel(x, dim: int = 1):
    """
    Adds a bias channel useful during pooling + whitening operation.
    """
    bias_size = list(x.size())
    bias_size[dim] = 1
    one = x.new_ones(bias_size)
    return torch.cat((x, one), dim)


# Credits: Matthijs Douze
def flatten(x: torch.Tensor, keepdims: bool = False):
    """
    Flattens B C H W input to B C*H*W output, optionally retains trailing dimensions.
    """
    y = x.view(x.size(0), -1)
    if keepdims:
        for _ in range(y.dim(), x.dim()):
            y = y.unsqueeze(-1)
    return y


def get_average_gem(
    activation_maps: List[torch.Tensor],
    p: int = 3,
    eps: float = 1e-6,
    clamp: bool = True,
    add_bias: bool = False,
    keepdims: bool = False,
):
    """
    Average Gem pooling of list of tensors. See #gem below for more information.

    Returns:
        x (torch.Tensor): Gem pooled tensor
    """
    activation_maps = torch.stack(
        [
            gem(
                activation_map,
                p,
                eps,
                clamp,
                add_bias,
                keepdims,
            )
            for activation_map in activation_maps
        ]
    )

    return torch.mean(activation_maps, dim=0)


# Credits: Matthijs Douze
def gem(
    x: torch.Tensor,
    p: int = 3,
    eps: float = 1e-6,
    clamp: bool = True,
    add_bias: bool = False,
    keepdims: bool = False,
):
    """
    Gem pooling on the given tensor.

    Args:
        x (torch.Tensor): tensor on which the pooling should be done
        p (int): pooling number.
                 If p=inf then simply perform max_pool2d
                 If p=1 and x tensor has grad, simply perform avg_pool2d
                 else, perform Gem pooling for specified p
        eps (float): if clamping the x tensor, use the eps for clamping
        clamp (float): whether to clamp the tensor
        add_bias (bool): whether to add the biad channel
        keepdims (bool): whether to flatten or keep the dimensions as is

    Returns:
        x (torch.Tensor): Gem pooled tensor
    """
    if p == math.inf or p == "inf":
        x = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    elif p == 1 and not (torch.is_tensor(p) and p.requires_grad):
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
    else:
        if clamp:
            x = x.clamp(min=eps)
        x = F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
    if add_bias:
        x = add_bias_channel(x)
    if not keepdims:
        x = flatten(x)
    return x


# Credits: Matthijs Douze
def l2n(x: torch.Tensor, eps: float = 1e-6, dim: int = 1):
    """
    L2 normalize the input tensor along the specified dimension

    Args:
        x (torch.Tensor): the tensor to normalize
        eps (float): epsilon to use to normalize to avoid the inf output
        dim (int): along which dimension to L2 normalize

    Returns:
        x (torch.Tensor): L2 normalized tensor
    """
    x = x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps).expand_as(x)
    return x


# Credits: Matthijs Douze
class MultigrainResize(transforms.Resize):
    """
    Resize with a `largest=False` argument
    allowing to resize to a common largest side without cropping
    Approach used in the Multigrain paper https://arxiv.org/pdf/1902.05509.pdf
    """

    def __init__(self, size: int, largest: bool = False, **kwargs):
        super().__init__(size, **kwargs)
        self.largest = largest

    @staticmethod
    def target_size(w: int, h: int, size: int, largest: bool = False):
        if (h < w) == largest:
            w, h = size, int(size * h / w)
        else:
            w, h = int(size * w / h), size
        size = (h, w)
        return size

    def __call__(self, img):
        size = self.size
        w, h = img.size
        target_size = self.target_size(w, h, size, self.largest)
        return TF.resize(img, target_size, self.interpolation)

    def __repr__(self):
        r = super().__repr__()
        return r[:-1] + f", largest={self.largest})"


# Credits: Matthijs Douze
class WhiteningTrainingImageDataset:
    """
    A set of training images for whitening
    """

    def __init__(self, base_dir: str, image_list_file: str, num_samples: int = 0):
        with g_pathmgr.open(image_list_file) as fopen:
            self.image_list = fopen.readlines()
        if num_samples > 0:
            self.image_list = self.image_list[:num_samples]
        self.root = base_dir
        self.N_images = len(self.image_list)
        logging.info(f"Loaded whitening data: {self.N_images}...")

    def get_num_images(self):
        return self.N_images

    def get_filename(self, i: int):
        return f"{self.root}/{self.image_list[i][:-1]}"


class InstreDataset:
    """
    A dataset class that reads and parses the Instre Dataset so it's ready to be used
    in the code for retrieval evaluations
    """

    def __init__(self, dataset_path: str, num_samples: int = 0):
        self.base_dir = dataset_path
        gnd_instre = scipy.io.loadmat(f"{self.base_dir}/gnd_instre.mat")
        self.gnd = gnd_instre["gnd"][0]
        self.qimlist = [fname[0] for fname in gnd_instre["qimlist"][0]]
        self.db_imlist = [fname[0] for fname in gnd_instre["imlist"][0]]

        if num_samples > 0:
            self.qimlist = self.qimlist[:num_samples]
            self.db_imlist = self.db_imlist[:num_samples]

        self.N_images = len(self.db_imlist)
        self.N_queries = len(self.qimlist)

        rs = np.random.RandomState(123)
        nq = self.N_queries
        self.val_subset = set(rs.choice(nq, nq // 10))
        logging.info(
            f"Loaded INSTRE dataset: {self.N_images}, queries: {self.N_queries}"
        )

    def get_num_images(self):
        """
        Number of images in the dataset
        """
        return self.N_images

    def get_num_query_images(self):
        """
        Number of query images in the dataset
        """
        return self.N_queries

    def get_filename(self, i: int):
        """
        Return the image filepath for the db image
        """
        return f"{self.base_dir}/{self.db_imlist[i]}"

    def get_query_filename(self, i: int):
        """
        Reutrn the image filepath for the query image
        """
        return f"{self.base_dir}/{self.qimlist[i]}"

    def get_query_roi(self, i: int):
        """
        INSTRE dataset has no notion of ROI so we return None.
        """
        return None

    def eval_from_ranks(self, ranks):
        """
        Return the mean average precision value or the train and validation both
        provided the ranks (scores of the model).
        """
        nq, nb = ranks.shape
        gnd = self.gnd
        sum_ap = 0
        sum_ap_val = 0
        for i in range(nq):
            positives = gnd[i][0][0] - 1
            ok = np.zeros(nb, dtype=bool)
            ok[positives] = True
            pos = np.where(ok[ranks[i]])[0]
            ap = score_ap_from_ranks_1(pos, len(positives))
            sum_ap += ap
            if i in self.val_subset:
                sum_ap_val += ap
        return sum_ap / nq, sum_ap_val / len(self.val_subset)

    def score(self, scores, verbose=True, temp_dir=None):
        """
        For the input scores of the model, calculate the AP metric
        """
        ranks = scores.argsort(axis=1)[:, ::-1]
        mAP, mAP_val = self.eval_from_ranks(ranks)
        if verbose:
            logging.info(f"INSTRE mAP={mAP} val {mAP_val}")
        return mAP, mAP_val


class RevisitedInstanceRetrievalDataset:
    """
    A dataset class used for the Revisited Instance retrieval datasets: Revisited
    Oxford and Revisited Paris. The object reads and parses the datasets so it's
    ready to be used in the code for retrieval evaluations.
    """

    def __init__(self, dataset: str, dir_main: str, num_samples=None):
        # Credits: https://github.com/filipradenovic/revisitop/blob/master/python/dataset.py#L6     # NOQA

        self.DATASETS = ["roxford5k", "rparis6k"]
        dataset = dataset.lower()
        assert is_revisited_dataset(dataset), f"Unknown dataset: {dataset}!"

        cfg = self.load_config(dir_main, dataset)
        cfg["ext"] = ".jpg"
        cfg["qext"] = ".jpg"

        cfg["dir_data"] = f"{dir_main}/{dataset}"
        cfg["dir_images"] = f"{cfg['dir_data']}/jpg"

        cfg["n"] = len(cfg["imlist"])
        cfg["nq"] = len(cfg["qimlist"])

        cfg["dataset"] = dataset

        self.cfg = cfg
        self.N_images = self.cfg["n"]
        self.N_queries = self.cfg["nq"]

        if num_samples is not None:
            self.N_queries = min(self.N_queries, num_samples)
            self.N_images = min(self.N_images, num_samples)

        logging.info(
            f"Dataset: {dataset}, images: {self.get_num_images()}, "
            f"queries: {self.get_num_query_images()}"
        )

    def load_config(self, dir_main, dataset):
        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = f"{dir_main}/{dataset}/gnd_{dataset}.pkl"
        cfg = load_file(gnd_fname)
        cfg["gnd_fname"] = gnd_fname

        return cfg

    def get_filename(self, i: int):
        """
        Return the image filepath for the db image
        """
        return f"{self.cfg['dir_images']}/{self.cfg['imlist'][i] + self.cfg['ext']}"

    def get_query_filename(self, i: int):
        """
        Reutrn the image filepath for the query image
        """
        return f"{self.cfg['dir_images']}/{self.cfg['qimlist'][i] + self.cfg['qext']}"

    def get_num_images(self):
        """
        Number of images in the dataset
        """
        return self.N_images

    def get_num_query_images(self):
        """
        Number of query images in the dataset
        """
        return self.N_queries

    def get_query_roi(self, i: int):
        """
        Get the ROI for the query image that we want to test retrieval
        """
        return self.cfg["gnd"][i]["bbx"]

    def score(self, sim, temp_dir=None):
        """
        For the input similarity scores of the model, calculate the mean AP metric
        and mean Precision@k metrics.
        """
        sim = sim.T
        # Credits: https://github.com/filipradenovic/revisitop/blob/master/python/example_evaluate.py  # NOQA
        ranks = np.argsort(-sim, axis=0)
        # revisited evaluation
        gnd = self.cfg["gnd"]
        # evaluate ranks
        ks = [1, 5, 10]

        # search for easy
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g["ok"] = np.concatenate([gnd[i]["easy"]])
            g["junk"] = np.concatenate([gnd[i]["junk"], gnd[i]["hard"]])
            gnd_t.append(g)

        mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

        # search for easy & hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g["ok"] = np.concatenate([gnd[i]["easy"], gnd[i]["hard"]])
            g["junk"] = np.concatenate([gnd[i]["junk"]])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

        # search for hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g["ok"] = np.concatenate([gnd[i]["hard"]])
            g["junk"] = np.concatenate([gnd[i]["junk"], gnd[i]["easy"]])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

        logging.info(
            ">> {}: mAP E: {}, M: {}, H: {}".format(
                self.cfg["dataset"],
                np.around(mapE * 100, decimals=2),
                np.around(mapM * 100, decimals=2),
                np.around(mapH * 100, decimals=2),
            )
        )
        logging.info(
            ">> {}: mP@k{} E: {}, M: {}, H: {}".format(
                self.cfg["dataset"],
                np.array(ks),
                np.around(mprE * 100, decimals=2),
                np.around(mprM * 100, decimals=2),
                np.around(mprH * 100, decimals=2),
            )
        )

        return {
            "mAP": {"e": mapE, "m": mapM, "h": mapH},
            "mp@k": {
                "k": ks,
                "e": mprE.tolist(),
                "m": mprM.tolist(),
                "h": mprH.tolist(),
            },
        }


# Credits: https://github.com/facebookresearch/deepcluster/blob/master/eval_retrieval.py    # NOQA
# Adapted by: Priya Goyal (prigoyal@fb.com)
class InstanceRetrievalImageLoader:
    """
    The custom loader for the Paris and Oxford Instance Retrieval datasets.
    """

    def __init__(self, S, transforms, center_crop):
        self.S = S
        self.transforms = transforms
        self.center_crop = center_crop

    def apply_img_transform(self, im):
        """
        Apply the pre-defined transforms on the image.
        """
        im_size_hw = np.array((im.size[1], im.size[0]))
        if self.S == -1:
            ratio = 1.0
        elif self.S == -2:
            if np.max(im_size_hw) > 124:
                ratio = 1024.0 / np.max(im_size_hw)
            else:
                ratio = -1
        else:
            ratio = float(self.S) / np.max(im_size_hw)
        new_size = tuple(np.round(im_size_hw * ratio).astype(np.int32))

        if not self.center_crop:
            # Center crop resizes image using pytorch transform,
            # which resizes smallest side, as opposed to largest side.
            im = im.resize((new_size[1], new_size[0]), Image.BILINEAR)

        im = self.transforms(im)

        return im, ratio

    def load_and_prepare_whitening_image(self, fname):
        """
        from the filename, load the whitening image and prepare it to be used by
        applying data transforms
        """
        with g_pathmgr.open(fname, "rb") as f:
            im = Image.open(f)
        if im.mode != "RGB":
            im = im.convert(mode="RGB")
        if self.transforms is not None:
            im = self.transforms(im)
        return im

    def load_and_prepare_instre_image(self, fname):
        """
        from the filename, load the db or query image and prepare it to be used by
        applying data transforms
        """
        with g_pathmgr.open(fname, "rb") as f:
            im = Image.open(f)
        if self.transforms is not None:
            im = self.transforms(im)
        return im

    def load_and_prepare_image(self, fname, roi=None):
        """
        Read image, get aspect ratio, and resize such as the largest side equals S.
        If there is a roi, adapt the roi to the new size and crop. Do not rescale
        the image once again. ROI format is (xmin,ymin,xmax,ymax)
        """
        # Read image, get aspect ratio, and resize such as the largest side equals S
        with g_pathmgr.open(fname, "rb") as f:
            img = Image.open(f).convert(mode="RGB")
        im_resized, ratio = self.apply_img_transform(img)
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            # ROI format is (xmin,ymin,xmax,ymax)
            roi = np.array(roi)
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[:, roi[1] : roi[3], roi[0] : roi[2]]
        return im_resized

    def load_and_prepare_revisited_image(self, img_path, roi=None):
        """
        Load the image, crop the roi from the image if the roi is not None,
        apply the image transforms.
        """
        # to avoid crashing for truncated (corrupted images)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)

        with g_pathmgr.open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        im_resized, ratio = self.apply_img_transform(img)
        # If there is a roi, adapt the roi to the new size and crop. Do not rescale
        # the image once again
        if roi is not None:
            # ROI format is (xmin,ymin,xmax,ymax)
            roi = np.array(roi)
            roi = np.round(roi * ratio).astype(np.int32)
            im_resized = im_resized[:, roi[1] : roi[3], roi[0] : roi[2]]
        return im_resized


class GenericInstanceRetrievalDataset:
    """
    A dataset class for reading images from a folder in the following simple format:
        /path/to/dataset
            - image_0.jpg
            ...
            - image_N.jpg

    The other datasets are in the process of being deprecated, currently this is
    available for use as a database or train split.
    """

    def __init__(
        self,
        data_path: str,
        num_samples: int = None,
    ):
        # Credits: https://github.com/filipradenovic/revisitop/blob/master/python/dataset.py#L6     # NOQA

        self.data_path = data_path

        self.query_filenames = None
        self.database_filenames = self._get_filenames(self.data_path)
        self.N_images = len(self.database_filenames)
        self.N_queries = None

        if num_samples is not None:
            self.N_images = min(self.N_images, num_samples)

        logging.info(f"Number of images: {self.get_num_images()}, ")

    def _get_filenames(self, data_path: str):
        fnames = []

        for fname in sorted(g_pathmgr.ls(data_path)):
            # Only put images in fnames.
            if not fname.endswith(".jpg"):
                continue

            full_fname = os.path.join(data_path, fname)
            fnames.append(full_fname)

        return np.array(fnames)

    def get_filename(self, i: int):
        """
        Return the image filepath for the db image
        """
        return self.database_filenames[i]

    def get_query_filename(self, i: int):
        """
        Rerurn the image filepath for the query image
        """
        logging.warning("GenericDataset does not yet have #get_query_filename support.")

        raise NotImplementedError

    def get_num_images(self):
        """
        Number of images in the dataset
        """
        return self.N_images

    def get_num_query_images(self):
        """
        Number of query images in the dataset
        """
        logging.warning("GenericDataset does not yet support query images.")

        raise NotImplementedError

    def get_query_roi(self, i: int):
        """
        GenericDataset does not yet have query_roi support
        """
        logging.warning("GenericDataset does not yet have query_roi support.")

        return None

    def score(self, sim, temp_dir=None):
        """
        For the input similarity scores of the model, calculate the mean AP metric
        and mean Precision@k metrics.
        """
        logging.warning("GenericDataset does not yet have #score support.")

        raise NotImplementedError


class InstanceRetrievalDataset:
    """
    A dataset class used for the Instance retrieval datasets:
    Oxford and Paris. The object reads and parses the datasets so it's
    ready to be used in the code for retrieval evaluations.

    Credits: https://github.com/facebookresearch/deepcluster/blob/master/eval_retrieval.py    # NOQA
    Adapted by: Priya Goyal (prigoyal@fb.com)
    """

    def __init__(self, path, eval_binary_path, num_samples=None):
        self.path = path
        self.eval_binary_path = eval_binary_path
        # Some images from the Paris dataset are corrupted. Standard practice is
        # to ignore them.
        # See: https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/corrupt.txt
        self.blacklisted_images = [
            "paris_louvre_000136",
            "paris_louvre_000146",
            "paris_moulinrouge_000422",
            "paris_museedorsay_001059",
            "paris_notredame_000188",
            "paris_pantheon_000284",
            "paris_pantheon_000960",
            "paris_pantheon_000974",
            "paris_pompidou_000195",
            "paris_pompidou_000196",
            "paris_pompidou_000201",
            "paris_pompidou_000467",
            "paris_pompidou_000640",
            "paris_sacrecoeur_000299",
            "paris_sacrecoeur_000330",
            "paris_sacrecoeur_000353",
            "paris_triomphe_000662",
            "paris_triomphe_000833",
            "paris_triomphe_000863",
            "paris_triomphe_000867",
        ]
        self.blacklisted = set(self.blacklisted_images)
        self.q_names = None
        self.q_index = None
        self.N_images = None
        self.N_queries = None
        self.q_roi = None
        self.load(num_samples=num_samples)

    def get_num_images(self):
        """
        Number of images in the dataset
        """
        return self.N_images

    def get_num_query_images(self):
        """
        Number of query images in the dataset
        """
        return self.N_queries

    def load(self, num_samples=None):
        """
        Load the data ground truth and parse the data so it's ready to be used.
        """
        # Load the dataset GT
        self.lab_root = f"{self.path}/lab/"
        self.img_root = f"{self.path}/jpg/"
        logging.info(f"Loading data: {self.path}")
        lab_filenames = np.sort(g_pathmgr.ls(self.lab_root))
        # Get the filenames without the extension
        self.img_filenames = [
            e[:-4]
            for e in np.sort(g_pathmgr.ls(self.img_root))
            if e[:-4] not in self.blacklisted
        ]

        # Parse the label files. Some challenges as filenames do not correspond
        # exactly to query names. Go through all the labels to:
        # i) map names to filenames and vice versa
        # ii) get the relevant regions of interest of the queries,
        # iii) get the indexes of the dataset images that are queries
        # iv) get the relevants / non-relevants list
        self.relevants = {}
        self.junk = {}
        self.non_relevants = {}

        self.filename_to_name = {}
        self.name_to_filename = OrderedDict()
        self.q_roi = {}
        for e in lab_filenames:
            if e.endswith("_query.txt"):
                q_name = e[: -len("_query.txt")]
                with g_pathmgr.open(f"{self.lab_root}/{e}") as fopen:
                    q_data = fopen.readline().split(" ")
                if q_data[0].startswith("oxc1_"):
                    q_filename = q_data[0][5:]
                else:
                    q_filename = q_data[0]
                self.filename_to_name[q_filename] = q_name
                self.name_to_filename[q_name] = q_filename
                with g_pathmgr.open(f"{self.lab_root}/{q_name}_ok.txt") as fopen:
                    good = {e.strip() for e in fopen}
                with g_pathmgr.open(f"{self.lab_root}/{q_name}_good.txt") as fopen:
                    good = good.union({e.strip() for e in fopen})
                with g_pathmgr.open(f"{self.lab_root}/{q_name}_junk.txt") as fopen:
                    junk = {e.strip() for e in fopen}
                good_plus_junk = good.union(junk)
                self.relevants[q_name] = [
                    i
                    for i in range(len(self.img_filenames))
                    if self.img_filenames[i] in good
                ]
                self.junk[q_name] = [
                    i
                    for i in range(len(self.img_filenames))
                    if self.img_filenames[i] in junk
                ]
                self.non_relevants[q_name] = [
                    i
                    for i in range(len(self.img_filenames))
                    if self.img_filenames[i] not in good_plus_junk
                ]
                self.q_roi[q_name] = np.array(
                    [float(q) for q in q_data[1:]], dtype=np.float32
                )

        self.q_names = list(self.name_to_filename.keys())
        self.q_index = np.array(
            [self.img_filenames.index(self.name_to_filename[qn]) for qn in self.q_names]
        )

        self.N_images = len(self.img_filenames)
        self.N_queries = len(self.q_index)

        if num_samples is not None:
            self.N_queries = min(self.N_queries, num_samples)
            self.N_images = min(self.N_images, num_samples)

    def score(self, sim, temp_dir):
        """
        From the input similarity score, compute the mean average precision
        """
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        idx = np.argsort(sim, axis=1)[:, ::-1]
        maps = [
            self.score_rnk_partial(i, idx[i], temp_dir) for i in range(self.N_queries)
        ]
        for i in range(self.N_queries):
            logging.info("{0}: {1:.2f}".format(self.q_names[i], 100 * maps[i]))
        logging.info(20 * "-")
        logging.info("Mean: {0:.2f}".format(100 * np.mean(maps)))

    def score_rnk_partial(self, i, idx, temp_dir):
        """
        Compute the mean AP for a given single query
        """
        rnk = np.array(self.img_filenames[: self.N_images])[idx]

        with g_pathmgr.open(f"{temp_dir}/{self.q_names[i]}.rnk", "w") as f:
            f.write("\n".join(rnk) + "\n")

        cmd = (
            f"{self.eval_binary_path} {self.lab_root}{self.q_names[i]} "
            f"{temp_dir}/{self.q_names[i]}.rnk"
        )
        p = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        map_ = float(p.stdout.readlines()[0])
        p.wait()
        return map_

    def get_filename(self, i):
        """
        Return the image filepath for the db image
        """
        return os.path.normpath(
            "{0}/{1}.jpg".format(self.img_root, self.img_filenames[i])
        )

    def get_query_filename(self, i):
        """
        Reutrn the image filepath for the query image
        """
        return os.path.normpath(
            f"{self.img_root}/{self.img_filenames[self.q_index[i]]}.jpg"
        )

    def get_query_roi(self, i):
        """
        Get the ROI for the query image that we want to test retrieval
        """
        return self.q_roi[self.q_names[i]]


class CopyDaysDataset:
    """
    A dataset class used for the Copydays dataset.
    """

    query_splits = (
        ["original", "strong"]
        + ["jpegqual_%d" % i for i in [3, 5, 8, 10, 15, 20, 30, 50, 75]]
        + ["crops_%d" % i for i in [10, 15, 20, 30, 40, 50, 60, 70, 80]]
    )

    database_splits = ["original"]

    def __init__(
        self, data_path: str, num_samples: int = None, use_distractors: bool = False
    ):
        # Credits: https://github.com/filipradenovic/revisitop/blob/master/python/dataset.py#L6     # NOQA

        self.data_path = data_path

        self.query_filenames = self._get_filenames(
            os.path.join(self.data_path, "queries")
        )
        database_dir_name = (
            "database_and_distractors" if use_distractors else "database"
        )
        self.database_filenames = self._get_filenames(
            os.path.join(self.data_path, database_dir_name)
        )
        self.N_queries = len(self.query_filenames)
        self.N_images = len(self.database_filenames)

        if num_samples is not None:
            self.N_queries = min(self.N_queries, num_samples)
            self.N_images = min(self.N_images, num_samples)

        logging.info(
            f"Dataset: copydays, images: {self.get_num_images()}, "
            f"queries: {self.get_num_query_images()}"
        )

    def _get_filenames(self, data_path: str):
        fnames = []

        for fname in sorted(g_pathmgr.ls(data_path)):
            # Only put images in fnames.
            if not fname.endswith(".jpg"):
                continue

            full_fname = os.path.join(data_path, fname)
            fnames.append(full_fname)

        return np.array(fnames)

    def get_filename(self, i: int):
        """
        Return the image filepath for the db image
        """
        return self.database_filenames[i]

    def get_query_filename(self, i: int):
        """
        Rerurn the image filepath for the query image
        """
        return self.query_filenames[i]

    def get_num_images(self):
        """
        Number of images in the dataset
        """
        return self.N_images

    def get_num_query_images(self):
        """
        Number of query images in the dataset
        """
        return self.N_queries

    def get_query_roi(self, i: int):
        """
        Copydays has no concept of ROI.
        """
        return None

    def score(self, sim, temp_dir=None):
        """
        For the input similarity scores of the model, calculate the mean AP metric
        and mean Precision@k metrics.
        """
        # Map at rank K
        ks = [1, 5, 10]

        query_filenames = self.query_filenames[0 : sim.shape[0]]
        database_filenames = self.database_filenames[0 : sim.shape[1]]

        # Calculate map for each query split
        results = {}
        for query_split in self.query_splits:

            # Get indeces of split queries.
            query_indeces = []
            for i, query_split_filename in enumerate(query_filenames):
                if query_split in query_split_filename:
                    query_indeces.append(i)

            # No queries for this split. Used for DEBUG_MODE when imposing data limit.
            if len(query_indeces) == 0:
                continue

            # Choose only rows of the split.
            query_split_filenames = query_filenames[query_indeces]
            split_sim = sim[query_indeces].T

            # Calculate the ranks.
            ranks = np.argsort(-split_sim, axis=0)

            has_query_match = False
            query_matches = []
            # Find matching database image for each query.
            for query_filename in query_split_filenames:
                matching_indices = []
                for i, database_filename in enumerate(database_filenames):
                    if self._is_query_database_match(database_filename, query_filename):
                        has_query_match = True
                        matching_indices.append(i)

                matches = {"ok": np.array(matching_indices)}
                query_matches.append(matches)

            # No database matches to compute mAP. Used in DEBUG_MODE when imposing data limit.
            if not has_query_match:
                continue

            # Compute macro average precision
            map_metric, _, mpr, _ = compute_map(ranks, query_matches, ks)

            results[query_split] = {
                "mAP": map_metric,
                "mp@k": {"k": ks, "mAP": mpr.tolist()},
            }

        return results

    def _is_query_database_match(self, database_filepath: str, query_filepath: str):
        """
        In the copydays dataset, the labels are based on the filename.
        e.g. for query with filename: 200000.jpg, the database filename
        will be one of: (200000.jpg, 200001.jpg, ..., 200099.jpg).
        """
        # Distractor images by definition do not match queries.
        # Distractors must have distractor in filename.
        database_filename = os.path.split(database_filepath)[-1]
        if "distractor" in database_filename:
            return False

        db_filename_number = self._find_image_number(database_filepath)
        query_filename_number = self._find_image_number(query_filepath)

        return (db_filename_number // 100) == (query_filename_number // 100)

    def _find_image_number(self, filename: str):
        filename_number = os.path.split(filename)[-1]
        filename_number = filename_number.split(".")[0].split("_")[-1]

        return int(filename_number)
