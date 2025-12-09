import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const modules = [
  {
    id: 1,
    title: 'Module 1: Foundations of Physical AI',
    duration: 'Weeks 1-2',
    description: 'Understand what makes Physical AI different from digital AI and why humanoid form matters in robotics.',
    keyTopics: ['Physics constraints in AI', 'Embodiment challenges', 'Real-world vs digital AI'],
  },
  {
    id: 2,
    title: 'Module 2: ROS 2 - The Robotic Nervous System',
    duration: 'Weeks 3-5',
    description: 'Master the "nervous system" of robots - how different software components communicate to create coordinated movement.',
    keyTopics: ['Nodes and topics', 'Services and actions', 'Message passing'],
  },
  {
    id: 3,
    title: 'Module 3: Simulation - Digital Twin Environments',
    duration: 'Weeks 6-7',
    description: 'Create virtual worlds where you can safely test robot behaviors and understand physics engines.',
    keyTopics: ['Gazebo simulation', 'Physics modeling', 'Sensor integration'],
  },
  {
    id: 4,
    title: 'Module 4: NVIDIA Isaac - AI-Robot Brain',
    duration: 'Weeks 8-10',
    description: 'Harness professional-grade tools for advanced vision and intelligent decision-making.',
    keyTopics: ['Isaac Sim', 'Perception systems', 'AI training in simulation'],
  },
  {
    id: 5,
    title: 'Module 5: Humanoid Control',
    duration: 'Weeks 11-12',
    description: 'Learn locomotion, balance control, manipulation and human-robot interaction.',
    keyTopics: ['Walking algorithms', 'Balance control', 'Manipulation'],
  },
  {
    id: 6,
    title: 'Module 6: Cognitive AI - Capstone Project',
    duration: 'Week 13',
    description: 'Build conversational AIs that process voice commands, plan navigation and manipulate objects.',
    keyTopics: ['Natural language processing', 'Task planning', 'Integration project'],
  },
];

export default function CourseOverview() {
  return (
    <section className={styles.courseOverview}>
      <div className="container">
        <div className="text--center padding-bottom--lg">
          <h2>Complete Course Outline: 13-Week Learning Journey</h2>
          <p className="padding-horiz--lg">
            A comprehensive curriculum designed to take you from beginner to advanced practitioner in Physical AI and Humanoid Robotics
          </p>
        </div>
        
        <div className="row">
          {modules.map((module) => (
            <div key={module.id} className="col col--4 padding-horiz--md padding-vert--sm">
              <div className={styles.moduleCard}>
                <div className={styles.moduleHeader}>
                  <h3>{module.title}</h3>
                  <span className={styles.duration}>{module.duration}</span>
                </div>
                
                <p>{module.description}</p>
                
                <div className={styles.keyTopics}>
                  <h4>Key Topics:</h4>
                  <ul>
                    {module.keyTopics.map((topic, idx) => (
                      <li key={idx}>{topic}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="text--center padding-vert--lg">
          <div className={styles.progressVisualization}>
            <div className={styles.progressBar}>
              <div className={styles.progressFill}></div>
            </div>
            <div className={styles.progressLabels}>
              <span>Beginner</span>
              <span>Intermediate</span>
              <span>Advanced</span>
              <span>Expert</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}