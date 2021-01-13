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
              <h3> Run locally </h3>
              <p>
                {' '}
                You can download the notebooks and source code to run the tutorials locally. You can modify the notebook to experiment with different settings. Remember to install pytorch, torchvision and
              vissl in the first cell of the notebook by running:{' '}
              </p>
              <MarkdownBlock>{bash`!pip install torch torchvision
!pip install 'git+https://github.com/facebookresearch/vissl.git@master'`}</MarkdownBlock>
            This installs the latest version of VISSL from github.
            </body>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
