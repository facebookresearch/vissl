/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');

const CWD = process.cwd();

const CompLibrary = require(`${CWD}/node_modules/docusaurus/lib/core/CompLibrary.js`);
const Container = CompLibrary.Container;
const MarkdownBlock = CompLibrary.MarkdownBlock;
const bash = (...args) => `~~~bash\n${String.raw(...args)}\n~~~`;
const TutorialSidebar = require(`${CWD}/core/TutorialSidebar.js`);

class TutorialHome extends React.Component {
  render() {
    return (
      <div className="docMainWrapper wrapper">
        <TutorialSidebar currentTutorialID={null} />
        <Container className="mainContainer documentContainer postContainer">
          <div className="post">
            <header className="postHeader">
              <h1 className="postHeaderTitle">Welcome to VISSL Tutorials</h1>
            </header>
            <body>
              <p>
                These tutorials will help you understand how to use VISSL from examples which are in the form of ipython notebooks.
              </p>
              <h3> Run tutorials interactively </h3>
              <p>
                Each tutorial can be run interactively in{' '}
                <a href="https://colab.research.google.com/notebooks/intro.ipynb">
                  {' '}
                  Google Colaboratory{' '}
                </a>{' '}
                which allows running the code directly in browser with access to GPUs. To run the tutorial in Colab, simply click
                on the button{' '}
                <strong>"Run in Google Colab"</strong> which looks like this:
              </p>
              <div className="tutorialButtonsWrapper">
              <div className="tutorialButtonWrapper buttonWrapper">
                <a className="tutorialButton button" target="_blank">
                  <img
                    className="colabButton"
                    align="left"
                    src="/img/colab_icon.png"
                  />
                  {'Run in Google Colab'}
                </a>
              </div>
            </div>
            <p>
              {' '}
              Every tutorial is standalone meaning that tutorial contain instructions for accessing data as well.
              At the start of every tutorial, the installation instructions are provided. We recommend to follow the tutorial steps
              to get started.{' '}
            </p>
            <h3> Run locally </h3>
            <p>
              {' '}
              There is also a button to download the notebook and source code to
              run it locally.{' '}
            </p>
            </body>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
