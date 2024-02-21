# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import re
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

from torch.utils.model_zoo import tqdm
from torchvision.datasets.utils import (
    check_integrity,
    download_file_from_google_drive,
    extract_archive,
)
from vissl.utils.io import makedir


def get_redirected_url(url: str):
    """
    Given a URL, returns the URL it redirects to or the
    original URL in case of no indirection
    """
    import requests

    with requests.Session() as session:
        with session.get(url, stream=True, allow_redirects=True) as response:
            if response.history:
                return response.url
            else:
                return url


def to_google_drive_download_url(view_url: str) -> str:
    """
    Utility function to transform a view URL of google drive
    to a download URL for google drive

    Example input:
        https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp/view

    Example output:
        https://drive.google.com/uc?export=download&id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp
    """
    splits = view_url.split("/")
    assert splits[-1] == "view"
    file_id = splits[-2]
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_google_drive_url(url: str, output_path: str, output_file_name: str):
    """
    Download a file from google drive

    Downloading an URL from google drive requires confirmation when
    the file of the size is too big (google drive notifies that
    anti-viral checks cannot be performed on such files)
    """
    import requests

    with requests.Session() as session:

        # First get the confirmation token and append it to the URL
        with session.get(url, stream=True, allow_redirects=True) as response:
            for k, v in response.cookies.items():
                if k.startswith("download_warning"):
                    url = url + "&confirm=" + v

        # Then download the content of the file
        with session.get(url, stream=True, verify=True) as response:
            makedir(output_path)
            path = os.path.join(output_path, output_file_name)
            total_size = int(response.headers.get("Content-length", 0))
            with open(path, "wb") as file:
                from tqdm import tqdm

                with tqdm(total=total_size) as progress_bar:
                    for block in response.iter_content(
                        chunk_size=io.DEFAULT_BUFFER_SIZE
                    ):
                        file.write(block)
                        progress_bar.update(len(block))


# The following methods are copied from torchvision, but we use g_pathmgr
# instead of `os` lib to support multiple distributed file systems.
def _get_google_drive_file_id(url: str) -> Optional[str]:
    parts = urlparse(url)

    if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
        return None

    match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)
    if match is None:
        return None

    return match.group("id")


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "vissl"})
        ) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def download_url(
    url: str,
    root: str,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
) -> None:
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under.
                                  If None, use the basename of the URL.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir(root)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
        return

    # expand redirect chain if needed
    url = get_redirected_url(url)

    # check if file is located on Google Drive
    file_id = _get_google_drive_file_id(url)
    if file_id is not None:
        return download_file_from_google_drive(file_id, root, filename, md5)

    # download the file
    try:
        print("Downloading " + url + " to " + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print(
                "Failed download. Trying https -> http instead."
                " Downloading " + url + " to " + fpath
            )
            _urlretrieve(url, fpath)
        else:
            raise e

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")


def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)
