import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

export default function CallToAction() {
  return (
    <section className={clsx('hero', styles.ctaSection)}>
      <div className="container">
        <div className="text--center">
          <h2 className={styles.title}>Ready to Build the Future of Robotics?</h2>
          <p className={styles.subtitle}>
            Start your journey to becoming a Physical AI and Humanoid Robotics expert today
          </p>
          
          <div className={styles.buttons}>
            <Link
              className="button button--primary button--lg"
              to="/docs/intro">
              Begin Learning
            </Link>
            
            <Link
              className="button button--secondary button--lg"
              to="/docs/module-01-foundations/01-intro-physical-ai">
              Explore Module 1
            </Link>
          </div>
          
          <div className={styles.benefits}>
            <div className="row">
              <div className="col col--4">
                <h3>13-Week Curriculum</h3>
                <p>Comprehensive learning path from basics to advanced concepts</p>
              </div>
              <div className="col col--4">
                <h3>Hands-On Practice</h3>
                <p>Real code examples and simulation exercises</p>
              </div>
              <div className="col col--4">
                <h3>Industry Tools</h3>
                <p>Learn with professional-grade software like ROS 2 and Isaac Sim</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}