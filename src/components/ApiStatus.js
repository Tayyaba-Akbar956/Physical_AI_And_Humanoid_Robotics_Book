import React, { useEffect, useState } from 'react';
import { API_URL } from '../utils/config';

export default function ApiStatus() {
    const [status, setStatus] = useState('checking');

    useEffect(() => {
        fetch(`${API_URL}/api/health`)
            .then(res => res.json())
            .then(() => setStatus('connected'))
            .catch(() => setStatus('disconnected'));
    }, []);

    return (
        <div className={`api-status ${status}`} style={{
            padding: '5px 10px',
            borderRadius: '4px',
            fontSize: '12px',
            display: 'inline-block',
            backgroundColor: status === 'connected' ? '#e6fffa' : status === 'disconnected' ? '#fff5f5' : '#f7fafc',
            color: status === 'connected' ? '#2c7a7b' : status === 'disconnected' ? '#c53030' : '#4a5568',
            border: `1px solid ${status === 'connected' ? '#81e6d9' : status === 'disconnected' ? '#feb2b2' : '#cbd5e0'}`,
            margin: '10px 0'
        }}>
            Backend: {status}
        </div>
    );
}
