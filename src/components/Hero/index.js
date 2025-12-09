import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

export default function Hero() {
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="row">
          <div className="col col--6 padding-vert--md">
            <h1 className="hero__title">Physical AI & Humanoid Robotics Textbook</h1>
            <p className="hero__subtitle">
              Bridging the gap between digital AI and physical embodiment
            </p>
            <p>
              The most comprehensive textbook that combines AI development with real-world robotics applications.
              Learn to create conversational humanoid robots that can navigate and interact with the physical world.
            </p>
            <div className={styles.buttons}>
              <Link
                className="button button--secondary button--lg"
                to="/docs/intro">
                Start Learning
              </Link>
            </div>
          </div>
          <div className="col col--6">
            <div className={styles.heroVisual}>
              <img 
                src="/img/hero-robot.png" 
                alt="Physical AI and Humanoid Robotics Visualization" 
                className={styles.heroImage}
              />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}