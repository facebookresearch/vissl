/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// See https://docusaurus.io/docs/site-config for all the possible
// site configuration options.

// Define this so it can be easily modified in scripts (to host elsewhere)
const baseUrl = "/";

// List of projects/orgs using your project for the users page.
const users = [];

const siteConfig = {
  title: 'VISSL', // Title for your website.
  tagline: 'A library for state-of-the-art self-supervised learning from images',
  url: 'https://vissl.ai', // Your website URL
  baseUrl: baseUrl, // Base URL for your project */
  // For github.io type URLs, you would set the url and baseUrl like:
  //   url: 'https://facebook.github.io',
  //   baseUrl: '/test-site/',

  // Used for publishing and more
  projectName: 'vissl',
  organizationName: 'facebookresearch',
  // For top-level user or org sites, the organization is still the same.
  // e.g., for the https://JoelMarcey.github.io site, it would be set like...
  //   organizationName: 'JoelMarcey'

  // For no header links in the top nav bar -> headerLinks: [],
  headerLinks: [
    {page: 'tutorials', label: 'Tutorials'},
    {href: "https://vissl.readthedocs.io/", label: 'Docs'},
    {href: "https://github.com/facebookresearch/vissl", label: 'GitHub'},
    { search: true } // position search box to the very right
  ],

  // If you have users set above, you add it here:
  users,

  /* path to images for header/footer */
  headerIcon: 'img/visslfavicon.png',
  //footerIcon: 'img/visslfavicon.png',
  favicon: 'img/visslfavicon.png',
  logo: 'img/Logo_Color_Light_BG.png',

  /* Colors for website */
  colors: {
    primaryColor: '#812CE5',
    secondaryColor: '#f868e6',
  },


  // This copyright info is used in /core/Footer.js and blog RSS/Atom feeds.
  copyright: `Copyright \u{00A9} ${new Date().getFullYear()} Facebook Inc`,

  highlight: {
    // Highlight.js theme to use for syntax highlighting in code blocks.
    theme: 'default',
  },

  // Add custom scripts here that would be placed in <script> tags.
  scripts: ['https://buttons.github.io/buttons.js'],

  // On page navigation for the current documentation page.
  onPageNav: 'separate',
  // No .html extensions for paths.
  cleanUrl: true,

  // Open Graph and Twitter card images.
  ogImage: 'img/vissllogo.svg',
  twitterImage: 'img/vissllogo.svg',

  // Google analytics
  gaTrackingId: 'UA-172675973-1',
  
  wrapPagesHTML: true,  

  // For sites with a sizable amount of content, set collapsible to true.
  // Expand/collapse the links and subcategories under categories.
  // docsSideNavCollapsible: true,

  // Show documentation's last contributor's name.
  // enableUpdateBy: true,

  // Show documentation's last update time.
  // enableUpdateTime: true,

  // You may provide arbitrary config keys to be used as needed by your
  // template. For example, if you need your repo's URL...
  //   repoUrl: 'https://github.com/facebook/test-site',
};

module.exports = siteConfig;
