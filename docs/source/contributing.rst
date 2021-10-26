Contributing to VISSL
==========================
We want to make contributing to this project as easy and transparent as possible.

Our Development Process
------------------------
Minor changes and improvements will be released on an ongoing basis. Larger changes (e.g., changesets implementing a new SSL approach, benchmark, new scaling feature etc) will be released on a more periodic basis.

Issues
------------------------
We use GitHub issues to track public bugs and questions. Please make sure to follow one of the
`issue templates <https://github.com/facebookresearch/vissl/issues/new/choose>`_
when reporting any issues.

Facebook has a `bounty program <https://www.facebook.com/whitehat/>`_ for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

Pull Requests
------------------------
We actively welcome your pull requests.

However, if you're adding any significant features (e.g. > 50 lines), please
make sure to have a corresponding issue to discuss your motivation and proposals,
before sending a PR. We do not always accept new features, and we take the following
factors into consideration:

1. Whether the same feature can be achieved without modifying VISSL. VISSL is designed to be extensible. As VISSL is a research library, we want to encourage modular components that are easy to extend. If some part is not as extensible as you would like, you can flag this.
2. Whether the feature is useful to a larger audience or only to a small portion of users.
3. Whether the proposed solution has a good design / interface.
4. Whether the proposed solution adds extra mental/practical overhead to users who don't need such a feature.
5. Whether the proposed solution breaks any existing APIs.

When sending a PR, please do:

1. Fork the repo and create your branch from :code:`main`.
2. If a PR contains multiple orthogonal changes, split it to several PRs.
3. If you've added code that should be tested, add tests.
4. If you've changed APIs, update the documentation.
5. Ensure the test suite passes. Follow `cpu test instructions <https://github.com/facebookresearch/vissl/blob/main/tests/README.md>`_ and `integration tests <https://github.com/facebookresearch/vissl/blob/main/dev/run_quick_tests.sh>`_.
6. Please run the `linter <https://github.com/facebookresearch/vissl/tree/main/dev#option-3-use-python-devlint_commitpy--a>`_.
7. Make sure your code follows our coding practices (see next section).
8. If you haven't already, complete the Contributor License Agreement ("CLA").

Coding Style
------------------------

Please follow `our coding practices <https://github.com/facebookresearch/vissl/blob/main/dev/README.md#practices-for-coding-quality>`_ and choose either option to properly format your code before submitting PRs.

Contributor License Agreement ("CLA")
------------------------------------------------
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA `here <https://code.facebook.com/cla>`_.

License
------------------------
By contributing to ssl_framework, you agree that your contributions will be licensed
under the `LICENSE <https://github.com/facebookresearch/vissl/blob/main/LICENSE>`_ file in the root directory of this source tree.
