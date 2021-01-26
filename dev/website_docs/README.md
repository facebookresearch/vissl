# Building VISSL website

Follow the instructions below to build vissl website

## Step1: Install dependencies

- Docusaurus requires yarn. Install yarn

```bash
brew install yarn
# or
curl -o- -L https://yarnpkg.com/install.sh | bash
```

- Install dependencies required to parse tutorials.

```bash
pip3 install nbformat==4.4.0 nbconvert==5.3.1 ipywidgets==7.5.1 tornado==4.2 bs4
```

**NOTE:** One can activate conda environment. No need to specify python version as long as you have python3.

## Step2: Build website

```bash
cd ~/vissl && ./dev/website_docs/build_website.sh
```

This will build the docusaurus website and run a script to parse the tutorials and generate:

`.html` files in the `website/_tutorials` folder
`.js` files in the `website/pages/tutorials` folder
`.py/.ipynb` files in the `website/static/files` folder

By the end of this, yarn will automatically start a local server where you can view the website.

## Step3: Build and publish the website

The following script will build the tutorials and the website and push to the `gh-pages` branch of github.com/facebookresearch/vissl.

```bash
./dev/website_docs/publish_website.sh
```

## Adding a new tutorial

The tutorials to include in the website are listed in `website/tutorials.json`. If you create a new tutorial add an entry to the list in this file. This is needed in order to generate the sidebar for the tutorials page.
