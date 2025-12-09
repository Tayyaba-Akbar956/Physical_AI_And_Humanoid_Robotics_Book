import React from 'react';
import styles from './styles.module.css';

const HeaderBackground = () => {
    // Generate random positions for particles
    // We'll use a fixed set to avoid hydration mismatch issues, but style them randomly
    const particles = [
        { type: 'circle', top: '10%', left: '5%', delay: '0s' },
        { type: 'square', top: '20%', left: '85%', delay: '2s' },
        { type: 'cross', top: '70%', left: '15%', delay: '1s' },
        { type: 'circuitLine', top: '10%', left: '50%', delay: '3s' },
        { type: 'circle', top: '80%', left: '90%', delay: '4s' },
        { type: 'square', top: '50%', left: '10%', delay: '1.5s' },
        { type: 'cross', top: '15%', left: '75%', delay: '2.5s' },
        { type: 'circuitLine', top: '60%', left: '30%', delay: '0.5s' },
        { type: 'circle', top: '40%', left: '40%', delay: '3.5s' },
        { type: 'square', top: '85%', left: '60%', delay: '1s' },
    ];

    return (
        <div className={styles.backgroundContainer}>
            {particles.map((p, index) => (
                <div
                    key={index}
                    className={`${styles.particle} ${styles[p.type]}`}
                    style={{
                        top: p.top,
                        left: p.left,
                        animationDelay: p.delay,
                    }}
                />
            ))}
        </div>
    );
};

export default HeaderBackground;
