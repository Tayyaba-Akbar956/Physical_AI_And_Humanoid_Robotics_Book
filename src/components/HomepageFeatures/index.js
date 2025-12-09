import React from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Module 1: Foundations',
    imageUrl: require('@site/static/img/module-1.png').default,
    description: (
      <>
        Understand what makes Physical AI different from digital AI and why
        humanoid form matters in robotics. Learn the unique challenges of
        applying AI to physical robots that must navigate the real world.
      </>
    ),
  },
  {
    title: 'Module 2: ROS 2 - The Robotic Nervous System',
    imageUrl: require('@site/static/img/module-2.png').default,
    description: (
      <>
        Master the "nervous system" of robots - how different software
        components communicate to create coordinated movement and sensing.
      </>
    ),
  },
  {
    title: 'Module 3: Simulation - Digital Twin Environments',
    imageUrl: require('@site/static/img/module-3.png').default,
    description: (
      <>
        Create virtual worlds where you can safely test robot behaviors
        and understand how physics engines simulate reality without expensive hardware.
      </>
    ),
  },
  {
    title: 'Module 4: NVIDIA Isaac - AI-Robot Brain',
    imageUrl: require('@site/static/img/module-4.png').default,
    description: (
      <>
        Harness professional-grade tools used by industry to add advanced
        vision and intelligent decision-making to robots.
      </>
    ),
  },
  {
    title: 'Module 5: Humanoid Control',
    imageUrl: require('@site/static/img/module-5.png').default,
    description: (
      <>
        Learn locomotion, balance control, manipulation and human-robot interaction
        for creating capable humanoid robots.
      </>
    ),
  },
  {
    title: 'Module 6: Cognitive AI',
    imageUrl: require('@site/static/img/module-6.png').default,
    description: (
      <>
        Build conversational AIs that can receive voice commands, process them,
        plan navigation and manipulate objects in the real world.
      </>
    ),
  },
];

function Feature({ imageUrl, title, description }) {
  return (
    <div className={clsx('col col--4', styles.featureCol)}>
      <div
        className={clsx(styles.featureCard)}
        role="region"
        aria-labelledby={`feature-heading-${title.replace(/\s+/g, '-').toLowerCase()}`}
        tabIndex="0"
      >
        <div className="text--center">
          {imageUrl && (
            <div className={styles.featureSvg}>
              <img
                src={imageUrl}
                className={styles.featureImage}
                alt={`${title} illustration`}
                role="img"
              />
            </div>
          )}
          <div className="padding-horiz--md">
            <Heading
              as="h3"
              id={`feature-heading-${title.replace(/\s+/g, '-').toLowerCase()}`}
            >
              {title}
            </Heading>
            <p>{description}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
        <div className="text--center padding-vert--md">
          <h2>Complete Course Outline: 13-Week Learning Journey</h2>
          <div className={styles.courseTimeline}>
            <div className={clsx('row', styles.timelineRow)}>
              <div className="col col--2"><strong>Weeks 1-2</strong></div>
              <div className="col col--10">Module 1: Foundations of Physical AI</div>
            </div>
            <div className={clsx('row', styles.timelineRow)}>
              <div className="col col--2"><strong>Weeks 3-5</strong></div>
              <div className="col col--10">Module 2: ROS 2 - The Robotic Nervous System</div>
            </div>
            <div className={clsx('row', styles.timelineRow)}>
              <div className="col col--2"><strong>Weeks 6-7</strong></div>
              <div className="col col--10">Module 3: Simulation - Digital Twin Environments</div>
            </div>
            <div className={clsx('row', styles.timelineRow)}>
              <div className="col col--2"><strong>Weeks 8-10</strong></div>
              <div className="col col--10">Module 4: NVIDIA Isaac - AI-Robot Brain</div>
            </div>
            <div className={clsx('row', styles.timelineRow)}>
              <div className="col col--2"><strong>Weeks 11-12</strong></div>
              <div className="col col--10">Module 5: Humanoid Control</div>
            </div>
            <div className={clsx('row', styles.timelineRow)}>
              <div className="col col--2"><strong>Week 13</strong></div>
              <div className="col col--10">Module 6: Cognitive AI - Capstone Project</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}