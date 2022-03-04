/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const MarkdownBlock = CompLibrary.MarkdownBlock; /* Used to read markdown */
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;
const bash = (...args) => `~~~bash\n${String.raw(...args)}\n~~~`;

class HomeSplash extends React.Component {
  render() {
    const {siteConfig, language = ''} = this.props;
    const {baseUrl, docsUrl} = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

    const SplashContainer = props => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = props => (
      <div className="splashLogo">
        <img src={props.img_src} alt="Project Logo" className="primaryLogoImage"/>
      </div>
    );

    const ProjectTitle = () => (
      <h2 className="projectTitle">
        <small>{siteConfig.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        <Logo img_src={siteConfig.logo} />
        <div className="inner">
          <ProjectTitle siteConfig={siteConfig} />
          <PromoSection>
            <Button href="#quickstart">Get Started</Button>
            <Button href={`${baseUrl}tutorials/`}>Tutorials</Button>
            <Button href={"https://vissl.readthedocs.io/en/v0.1.6/"}>Docs</Button>
            <Button href={"https://github.com/facebookresearch/vissl"}>GitHub</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

function SocialBanner() {
  return (
    <div className="SocialBannerWrapper">
      <div className="SocialBanner">
        Support Ukraine 🇺🇦{' '}
        <a href="https://opensource.fb.com/support-ukraine">
          Help Provide Humanitarian Aid to Ukraine
        </a>
        .
      </div>
    </div>
  );
}

function VideoContainer() {
  return (
    <div className="container text--center margin-bottom--xl">
      <div className="row">
        <div className="col" style={{textAlign: 'center'}}>
          <h2>Check it out in the intro video</h2>
          <div>
            <iframe
              width="560"
              height="315"
              src="https://www.youtube.com/embed/-0Bt-1ei7yw"
              title="Explain Like I'm 5: VISSL"
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
        </div>
      </div>
    </div>
  );
}

class Index extends React.Component {
  render() {
    const {config: siteConfig, language = ''} = this.props;
    const {baseUrl} = siteConfig;

    const Block = props => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );


    const Description = () => (
      <Block background="light">
        {[
          {
            content:
              'This is another description of how this project is useful',
            image: `${baseUrl}img/docusaurus.svg`,
            imageAlign: 'right',
            title: 'Description',
          },
        ]}
      </Block>
    );

    const pre = '```';

    const codeExample = `${pre}bash
python3 run_distributed_engines.py config=quick_1gpu_resnet50_simclr config.DATA.TRAIN.DATA_SOURCES=[synthetic]
    `;

    const downloadBlock = `${pre}bash
cd /tmp/ && mkdir -p /tmp/configs/config
wget -q -O configs/__init__.py https://dl.fbaipublicfiles.com/vissl/tutorials/configs/__init__.py
wget -q -O configs/config/quick_1gpu_resnet50_simclr.yaml https://dl.fbaipublicfiles.com/vissl/tutorials/configs/quick_1gpu_resnet50_simclr.yaml
wget -q  https://dl.fbaipublicfiles.com/vissl/tutorials/run_distributed_engines.py
    `;

    const installBlock = `${pre}bash
conda create -n vissl python=3.8
conda activate vissl
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
conda install -c vissl -c iopath -c conda-forge -c pytorch -c defaults apex vissl
    `;

    const QuickStart = () => (
      <div
        className="productShowcaseSection"
        id="quickstart"
        style={{textAlign: 'center'}}>
        <h2>Get Started</h2>
        <Container>
          <ol>
            <li>
              <h4>Install VISSL:</h4>
              <a>via conda:</a>
              <MarkdownBlock>{installBlock}</MarkdownBlock>
            </li>
            <li>
              <h4>Download SimCLR yaml config and builtin distributed launcher:  </h4>
              <MarkdownBlock>{downloadBlock}</MarkdownBlock>
            </li>
            <li>
              <h4>Try training SimCLR model on 1-gpu:  </h4>
              <MarkdownBlock>{codeExample}</MarkdownBlock>
            </li>
          </ol>
        </Container>
      </div>
    );


    const Features = () => (
    <div className="productShowcaseSection" style={{textAlign: 'center'}}>
      <Block layout="fourColumn">
        {[
          {
            content:
              'Built on top of PyTorch which allows using all of its components.',
            image: `${baseUrl}img/pytorch.svg`,
            imageAlign: 'top',
            title: 'Powered by PyTorch',
          },
          {
            content:
              'Reproducible reference implementation of SOTA self-supervision approaches (like SimCLR, MoCo, PIRL, SwAV etc) and their components that can be reused. Also supports supervised trainings.',
            image: `${baseUrl}img/ssl_approaches.svg`,
            imageAlign: 'top',
            title: 'SOTA Self-Supervision methods',
          },
          {
            content:
              'Variety of benchmarks tasks (linear image classification, full finetuning, semi-supervised, low-shot, nearest neighbor, object detection) available to evaluate models.',
            image: `${baseUrl}img/benchmark_suite.svg`,
            imageAlign: 'top',
            title: 'Benchmark tasks',
          },
          {
            content:
              'Easy to train model on 1-gpu, multi-gpu and multi-node. Seamless scaling to large scale data and model sizes with FP16, LARC etc',
            image: `${baseUrl}img/scalable.svg`,
            imageAlign: 'top',
            title: 'Scalable',
          },
        ]}
      </Block>
    </div>
  );

    const Showcase = () => {
      if ((siteConfig.users || []).length === 0) {
        return null;
      }

      const showcase = siteConfig.users
        .filter(user => user.pinned)
        .map(user => (
          <a href={user.infoLink} key={user.infoLink}>
            <img src={user.image} alt={user.caption} title={user.caption} />
          </a>
        ));

      const pageUrl = page => baseUrl + (language ? `${language}/` : '') + page;

      return (
        <div className="productShowcaseSection paddingBottom">
          <h2>Who is Using This?</h2>
          <p>This project is used by all these people</p>
          <div className="logos">{showcase}</div>
          <div className="more-users">
            <a className="button" href={pageUrl('users.html')}>
              More {siteConfig.title} Users
            </a>
          </div>
        </div>
      );
    };

    return (
      <div>
        <SocialBanner />
        <HomeSplash siteConfig={siteConfig} language={language} />
        <VideoContainer />
        <div className="mainContainer">
          <Features />
          <QuickStart />
        </div>
      </div>
    );
  }
}

module.exports = Index;
