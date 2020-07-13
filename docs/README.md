# Generating Sphinx docs for VISSL

## Setup

First Install Sphinx:

```bash
pip install -U sphinx
# Install the dependencies like mobile friendly [`sphinx_rtd_theme` style theme](here https://github.com/rtfd/sphinx_rtd_theme)
cd docs && pip install -r requirements.txt
```

If setting up the sphinx doc for the first time for a project, run
```bash
cd docs && sphinx-quickstart
```

This will create the `conf.py` template in `source`. You can customize it for your needs. See VISSL's `conf.py` as an example.

## Build docs

```bash
cd docs && make html
```
Now you can see the generated html `index.html` under `build/html/`. Send a PR.

## View html pages
As you make changes to the doc, you can view the updated html docs. For this, tart a simple python server:

```bash
# you can pick whatever port you want to use
python -m http.server 8097
```
Navigate to: http://0.0.0.0:8097/
