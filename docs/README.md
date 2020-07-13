# Generating Sphinx docs for VISSL

## Setup

First Install Sphinx:

```bash
pip install -U sphinx
# Install the dependencies like mobile friendly [`sphinx_rtd_theme` style theme](here https://github.com/rtfd/sphinx_rtd_theme)
cd docs && pip install -r requirements.txt
```

If setting up the sphinx doc for the first time, run
```bash
cd docs && sphinx-quickstart
```

This will create the `conf.py` template in `source`. You can customize it for your needs. See VISSL's `conf.py` as an example.

## Build docs

```bash
cd docs && make html
```
Now you can see the generated html `index.html` under `build/html/`. Send PR.
