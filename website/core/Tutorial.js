/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * @format
 */

const React = require('react');

const fs = require('fs-extra');
const path = require('path');
const CWD = process.cwd();

const CompLibrary = require(`${CWD}/node_modules/docusaurus/lib/core/CompLibrary.js`);
const Container = CompLibrary.Container;

const TutorialSidebar = require(`${CWD}/core/TutorialSidebar.js`);

function renderDownloadIcon() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      data-prefix="fas"
      data-icon="file-download"
      className="svg-inline--fa fa-file-download fa-w-12"
      role="img"
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 384 512">
      <path
        fill="currentColor"
        d="M224 136V0H24C10.7 0 0 10.7 0 24v464c0 13.3 10.7 24 24 24h336c13.3 0 24-10.7 24-24V160H248c-13.2 0-24-10.8-24-24zm76.45 211.36l-96.42 95.7c-6.65 6.61-17.39 6.61-24.04 0l-96.42-95.7C73.42 337.29 80.54 320 94.82 320H160v-80c0-8.84 7.16-16 16-16h32c8.84 0 16 7.16 16 16v80h65.18c14.28 0 21.4 17.29 11.27 27.36zM377 105L279.1 7c-4.5-4.5-10.6-7-17-7H256v128h128v-6.1c0-6.3-2.5-12.4-7-16.9z"
      />
    </svg>
  );
}

class Tutorial extends React.Component {
  render() {
    const {baseUrl, tutorialID} = this.props;

    const htmlFile = `${CWD}/_tutorials/${tutorialID}.html`;
    const normalizedHtmlFile = path.normalize(htmlFile);

    return (
      <div className="docMainWrapper wrapper">
        <TutorialSidebar currentTutorialID={tutorialID} />
        <Container className="mainContainer">
          <div className="tutorialButtonsWrapper">
          <div className="tutorialButtonWrapper buttonWrapper">
              <a
                className="tutorialButton button"
                download
                href={`https://colab.research.google.com/github/facebookresearch/vissl/blob/v0.1.6/tutorials/${tutorialID}.ipynb`}
                target="_blank">
                <img
                  className="colabButton"
                  align="left"
                  src={`${baseUrl}img/colab_icon.png`}
                />
                {'Run in Google Colab'}
              </a>
            </div>
            <div className="tutorialButtonWrapper buttonWrapper">
              <a
                className="tutorialButton button"
                download
                href={`${baseUrl}files/${tutorialID}.ipynb`}
                target="_blank">
                {renderDownloadIcon()}
                {'Download Tutorial Jupyter Notebook'}
              </a>
            </div>
            <div className="tutorialButtonWrapper buttonWrapper">
              <a
                className="tutorialButton button"
                download
                href={`${baseUrl}files/${tutorialID}.py`}
                target="_blank">
                {renderDownloadIcon()}
                {'Download Tutorial Source Code'}
              </a>
            </div>
          </div>
          <div
            className="tutorialBody"
            dangerouslySetInnerHTML={{
              __html: fs.readFileSync(normalizedHtmlFile, {encoding: 'utf8'}),
            }}
          />
        </Container>
      </div>
    );
  }
}

module.exports = Tutorial;
