"""PyTorch Dataset class for PKL+HDF format slides."""
import io
import os
import pickle
import random

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF


class HDFSlideDataset(torch.utils.data.Dataset):

    def __init__(self, pkl_path_dict, patch_sizes=[1024, 4096, 16384],
                 index_by="slide", transform=None, **kwargs):
        super().__init__(**kwargs)

        # Load all pickle data and merge them into one dict
        assert isinstance(pkl_path_dict, dict)
        self._pkl_path_dict = pkl_path_dict
        self._load_pkl_files()

        # Register the patch sizes to load
        assert isinstance(patch_sizes, list)
        self._patch_sizes = patch_sizes
        for patch_size in patch_sizes:
            assert patch_size in (1024, 4096, 16384)

        # Build a list of data items to index on __getitem__
        # slide: each item yields a random patch from the slide
        # patch: each item refers to a specific patch
        assert index_by in ("slide", "patch")
        self._index_by = index_by
        self._build_data_list()

        # Register transforms
        self.transform = transform

    def _load_pkl_files(self):
        self._pkl_data = {}
        for h5_folder, pkl_path in self._pkl_path_dict.items():
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
                for key in pkl_data.keys():
                    h5_path = os.path.join(
                        os.path.dirname(pkl_path),
                        h5_folder,
                        str(pkl_data[key]["uuid"]) + ".h5",
                    )
                    assert os.path.isfile(h5_path), f"File not found at: {h5_path}"
                    pkl_data[key]["h5_path"] = h5_path

                self._pkl_data.update(pkl_data)

    def _build_data_list(self):
        data_list = []
        if self._index_by == "slide":
            for slide_key, slide_data in self._pkl_data.items():
                sub_list = []
                for patch_size in self._patch_sizes:
                    for sub_idx in range(slide_data[f"num_patches_{patch_size}"]):
                        sub_list.append((patch_size, sub_idx))
                data_item = (slide_key, sub_list)
                data_list.append(data_item)

        elif self._index_by == "patch":
            for slide_key, slide_data in self._pkl_data.items():
                for patch_size in self._patch_sizes:
                    for sub_idx in range(slide_data[f"num_patches_{patch_size}"]):
                        data_item = (slide_key, patch_size, sub_idx)
                        data_list.append(data_item)

        else:
            raise NotImplementedError

        self._data_list = data_list

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, data_idx):
        # Define the data item
        if self._index_by == "slide":
            slide_key, sub_list = self._data_list[data_idx]
            patch_size, patch_idx = random.choice(sub_list)
        elif self._index_by == "patch":
            slide_key, patch_size, patch_idx = self._data_list[data_idx]
        else:
            raise NotImplementedError

        # Load image
        h5_path = self._pkl_data[slide_key]["h5_path"]
        h5_slide = HDFSlide(h5_path)
        image = h5_slide.get_predefined_patch(patch_idx, patch_size)

        # Apply PyTorch routines
        if self.transform:
            image = self.transform(image)
        return TF.to_tensor(image)


PKL_PATH_BASE = "/storage5/pp/share/210829_extracted_hdf_files/"


class HDFSlideDataset_Train(HDFSlideDataset):

    def __init__(
        self,
        pkl_path_dict={
            "io": PKL_PATH_BASE + "wsi_info_final_io_cell-tissue_trn.pkl",
            "pdl1": PKL_PATH_BASE + "wsi_info_final_pdl1_cell_trn.pkl",
        },
        **kwargs,
    ):
        super().__init__(pkl_path_dict, **kwargs)


class HDFSlideDataset_Validation(HDFSlideDataset):

    def __init__(
        self,
        pkl_path_dict={
            "io": PKL_PATH_BASE + "wsi_info_final_io_cell-tissue_val.pkl",
            "pdl1": PKL_PATH_BASE + "wsi_info_final_pdl1_cell_val.pkl",
        },
        **kwargs,
    ):
        super().__init__(pkl_path_dict, **kwargs)


class HDFSlide:
    """Class for accessing a single HDF format slide."""

    def __init__(self, fpath):
        self._fpath = fpath
        self._handle = h5py.File(self._fpath, 'r')

    def _retrieve_patch(self, ds_name, patch_idx):
        bgr_image = cv2.imdecode(self._handle[ds_name][patch_idx], cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def get_predefined_patch(self, patch_idx, patch_size_reference):
        """Simply retrieve and decode a pre-defined patch."""
        assert patch_size_reference in (1024, 4096, 16384)
        ds_name = f'patches_{patch_size_reference}'
        return self._retrieve_patch(ds_name, patch_idx)

    def get_random_predefined_patch(self, patch_size_reference):
        """Retrieve and decode a randomly selected pre-defined patch."""
        assert patch_size_reference in (1024, 4096, 16384)
        ds_name = f'patches_{patch_size_reference}'
        dataset = self._handle[ds_name]
        rand_idx = np.random.randint(0, dataset.len())
        return self._retrieve_patch(ds_name, rand_idx)

    def get_unaligned_patch(self, x_start_original, y_start_original, patch_size_reference):
        """Get a randomly defined patch by retrieving 4 nearest patches.
        Parameters
        ----------
        x_start_original: int
            Left-most coordinate of the requested patch, at original WSI MPP.
        y_start_original: int
            Top-most coordinate of the requested patch, at original WSI MPP.
        patch_size_reference: int
            The requested patch size at reference MPP. Note that irrespective of
            this value, all retrieved patches are 1024x1024 in size.
        """
        assert patch_size_reference in (1024, 4096, 16384)
        ds_name = f"patches_{patch_size_reference}"
        dataset = self._handle[ds_name]
        metadata = pickle.load(io.BytesIO(dataset.attrs['metadata'].tobytes()))

        # Determine the floating point grid position of the requested patch
        offset_x_original = metadata["offset_x_original"]
        offset_y_original = metadata["offset_y_original"]
        patch_size_original = metadata["patch_size_original"]
        x_start_grid = (x_start_original - offset_x_original) / patch_size_original
        y_start_grid = (y_start_original - offset_y_original) / patch_size_original

        def get_grid(x_idx, y_idx, fill_value=255):
            """Retrieve the image patch at the specified grid position.
            If the patch does not exist, it returns an image with constant values.
            """
            assert isinstance(x_idx, int) and isinstance(y_idx, int)
            if (x_idx, y_idx) in metadata["grid_to_idx_map"]:
                patch_idx = metadata["grid_to_idx_map"][(x_idx, y_idx)]
                return self._retrieve_patch(ds_name, patch_idx)
            else:
                return np.full((patch_size, patch_size, 3), dtype=np.uint8,
                               fill_value=fill_value)

        # Fill a quad-patch with relevant pixels
        patch_size = 1024  # all patches are assumed to be stored at this size
        quad_patch = np.empty((2 * patch_size, 2 * patch_size, 3), dtype=np.uint8)
        quad_patch[:patch_size, :patch_size, :] = \
            get_grid(int(x_start_grid), int(y_start_grid))
        quad_patch[patch_size:, :patch_size, :] = \
            get_grid(int(x_start_grid), int(np.ceil(y_start_grid)))
        quad_patch[:patch_size, patch_size:, :] = \
            get_grid(int(np.ceil(x_start_grid)), int(y_start_grid))
        quad_patch[patch_size:, patch_size:, :] = \
            get_grid(int(np.ceil(x_start_grid)), int(np.ceil(y_start_grid)))

        # Find actual required region from the quad-patch
        sub_x_pos = (x_start_grid - np.floor(x_start_grid)) * patch_size
        sub_y_pos = (y_start_grid - np.floor(y_start_grid)) * patch_size
        return cv2.getRectSubPix(quad_patch, (patch_size, patch_size),
                                 (sub_x_pos + 0.5 * patch_size,
                                  sub_y_pos + 0.5 * patch_size))


if __name__ == "__main__":

    output_dir = "test_dataset_output"
    os.makedirs(output_dir, exist_ok=True)

    # Extract a random patch
    h5_slide = HDFSlide("/storage5/pp/share/210829_extracted_hdf_files/pdl1/"
                        "00302c97-d1b0-4c0d-ab5a-0120407abaf8.h5")
    rgb_patch = h5_slide.get_unaligned_patch(
        x_start_original=23000,
        y_start_original=27000,
        patch_size_reference=4096,
    )
    cv2.imwrite(os.path.join(output_dir, "sample.jpg"), rgb_patch[:, :, ::-1])

    # Save random samples from the training set
    trn_dataset = HDFSlideDataset_Train()
    trn_dataloader = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
    )
    print(f"len(trn_dataset) = {len(trn_dataset)}")
    trn_batch = next(iter(trn_dataloader))
    for i in range(trn_batch.shape[0]):
        rgb_img = (trn_batch[i, :].permute([1, 2, 0]).numpy() * 255).astype(np.uint8)
        bgr_img = rgb_img[:, :, ::-1]
        cv2.imwrite(os.path.join(output_dir, f"trn_{i:02d}.jpg"), bgr_img)

    # Save random samples from the validation set
    val_dataset = HDFSlideDataset_Validation()
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
    )
    print(f"len(val_dataset) = {len(val_dataset)}")
    val_batch = next(iter(val_dataloader))
    for i in range(val_batch.shape[0]):
        rgb_img = (val_batch[i, :].permute([1, 2, 0]).numpy() * 255).astype(np.uint8)
        bgr_img = rgb_img[:, :, ::-1]
        cv2.imwrite(os.path.join(output_dir, f"val_{i:02d}.jpg"), bgr_img)