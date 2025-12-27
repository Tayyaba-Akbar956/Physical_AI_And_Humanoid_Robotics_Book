// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking

const { themes } = require('prism-react-renderer');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'Bridging the gap between digital AI and physical embodiment',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://physical-ai-book.vercel.app',
  // Set the /<baseUrl>/ pathname under which your site is served
  baseUrl: '/',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/Tayyaba-Akbar956/Physical_AI_And_Humanoid_Robotics_Book/tree/main/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themes: [
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        hashed: true,
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/logo.svg',
      navbar: {
        title: 'Physical AI Textbook',
        logo: {
          alt: 'Physical AI & Humanoid Robotics Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Textbook',
          },
          {
            href: 'https://github.com/Tayyaba-Akbar956/Physical_AI_And_Humanoid_Robotics_Book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Content',
            items: [
              {
                label: 'Textbook',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/Tayyaba-Akbar956/Physical_AI_And_Humanoid_Robotics_Book',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Contribute',
                href: 'https://github.com/Tayyaba-Akbar956/Physical_AI_And_Humanoid_Robotics_Book',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
      },
      prism: {
        theme: themes.github,
        darkTheme: themes.dracula,
      },
    }),
  plugins: [
    // Add the chatbot widget to the site
    async function chatbotPlugin(context, options) {
      return {
        name: 'chatbot-plugin',
        injectHtmlTags() {
          return {
            postBodyTags: [
              {
                tagName: 'script',
                attributes: {
                  src: '/frontend/rag-widget/embed-script.js',
                  type: 'text/javascript',
                  async: true,
                },
              },
            ],
          };
        },
      };
    },
  ],
};

module.exports = config;