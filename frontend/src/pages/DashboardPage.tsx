import { useState, useEffect } from 'react';
import React from 'react';

export default function DashboardPage() {
  const [tableauUrl, setTableauUrl] = useState<string>('');

  useEffect(() => {
    const url = import.meta.env.VITE_TABLEAU_REPORT_URL || '';
    setTableauUrl(url);
  }, []);

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Portfolio Analytics Dashboard</h1>
        <div style={{ color: 'var(--text-secondary)' }}>Live connection to Tableau Public</div>
      </div>
      
      <div className="page-content" style={{ display: 'flex', flexDirection: 'column' }}>
        
        <div className="card" style={{ flex: 1, minHeight: 600, padding: '1rem', display: 'flex' }}>
          
          {!tableauUrl ? (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: 'var(--bg-color)', borderRadius: '0.5rem', border: '1px dashed var(--border-color)' }}>
              <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem' }}>Tableau Integration Required</h2>
              <p style={{ color: 'var(--text-secondary)', maxWidth: 450, textAlign: 'center', marginBottom: '2rem' }}>
                The application is ready to embed your Portfolio-wide Analytics. 
                Please provide your Tableau Public URL via the <code>.env</code> file to visually render the live dashboard.
              </p>
              <div style={{ backgroundColor: '#1E293B', color: '#FFF', padding: '1.5rem', borderRadius: '0.5rem', fontFamily: 'monospace', fontSize: '0.875rem', width: '100%', maxWidth: 650 }}>
                # .env.local<br/>
                VITE_TABLEAU_REPORT_URL=https://public.tableau.com/views/YourReport/Dashboard1
              </div>
            </div>
          ) : (
            <div style={{ width: '100%', height: '100%', minHeight: '600px', overflow: 'hidden', borderRadius: '0.25rem' }}>
              {React.createElement('tableau-viz', {
                id: 'tableauViz',
                src: tableauUrl,
                toolbar: 'hidden',
                'hide-tabs': 'true',
                style: { width: '100%', height: '100%', minHeight: '600px' }
              })}
            </div>
          )}
          
        </div>
      </div>
    </>
  );
}
