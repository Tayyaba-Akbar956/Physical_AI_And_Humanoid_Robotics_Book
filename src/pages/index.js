import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import Heading from '@theme/Heading';
import styles from './index.module.css';
import HeaderBackground from '@site/src/components/HeaderBackground';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={styles.heroBanner}>
      <HeaderBackground />
      <div className={styles.heroContainer}>
        {/* Left side: Custom robot image */}
        <div className={styles.leftSide}>
          <img
            src={useBaseUrl('/img/hero-robot.png')}
            alt="Robot illustration for Physical AI & Humanoid Robotics Textbook"
            className={styles.robotImage}
          />
        </div>

        {/* Right side: Title, subtitle, and button */}
        <div className={styles.rightSide}>
          <Heading as="h1" className={styles.heroTitle}>
            Physical AI & Humanoid Robotics Textbook
          </Heading>
          <p className={styles.heroSubtitle}>Bridging the gap between digital AI and physical embodiment</p>
          <div className={styles.buttons}>
            <Link
              className={styles.startButton}
              to="/docs/intro">
              Start Reading
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Comprehensive textbook for Physical AI & Humanoid Robotics">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}