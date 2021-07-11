# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os


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
            os.makedirs(output_path, exist_ok=True)
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
