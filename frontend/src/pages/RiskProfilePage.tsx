import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts';

interface ShapData {
  feature: string;
  value: number;
}

interface RiskExplanation {
  explanation: string;
}

export default function RiskProfilePage() {
  const [searchParams] = useSearchParams();
  const id = searchParams.get('id') || '100001';
  const name = searchParams.get('name') || 'Maria Garcia';

  const [probability, setProbability] = useState<number>(82);
  const [shapValues, setShapValues] = useState<ShapData[]>([]);
  const [explanation, setExplanation] = useState<string>('');
  const [loadingExpl, setLoadingExpl] = useState(false);

  // What-If parameters
  const [income, setIncome] = useState<number>(45000);
  const [creditAmount, setCreditAmount] = useState<number>(250000);

  // Fetch initial profile data
  useEffect(() => {
    // Initial fetch of live ML prediction based on actual inputs
    fetchPrediction();
  }, [id]);

  const fetchPrediction = async () => {
    try {
      const resp = await fetch(`http://localhost:8000/predict/whatif?income=${income}&credit_amount=${creditAmount}`);
      if (resp.ok) {
        const data = await resp.json();
        setProbability(Math.round(data.prediction * 100));
        setShapValues(data.shap_values);
      }
    } catch (e) {
      console.error(e);
      // Fallback dummy
      setShapValues([
        { feature: 'Income', value: -0.12 },
        { feature: 'Credit Amount', value: 0.25 },
        { feature: 'Age', value: -0.05 },
        { feature: 'Employment Years', value: -0.18 }
      ]);
    }
  };

  const handleGenerateExplanation = async () => {
    setLoadingExpl(true);
    try {
      const resp = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          risk_score: probability, 
          shap_values: shapValues 
        })
      });
      const data: RiskExplanation = await resp.json();
      setExplanation(data.explanation);
    } catch (e) {
      setExplanation("Unable to connect to AI Explanation service.");
    }
    setLoadingExpl(false);
  };

  // SVG Gauge Calculations
  const radius = 120;
  const strokeWidth = 24;
  const normalizedProb = probability / 100;
  const dashArray = radius * Math.PI;
  const dashOffset = dashArray - dashArray * normalizedProb;
  
  // Dynamic color
  const color = probability > 60 ? 'var(--danger)' : probability > 30 ? 'var(--warning)' : 'var(--success)';

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">{name}'s Risk Profile</h1>
        <div style={{ color: 'var(--text-secondary)' }}>Application ID: #{id}</div>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(400px, 1fr) 2fr', gap: '2rem' }}>
        
        {/* Left Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          
          {/* Gauge Widget */}
          <div className="card" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <h3 style={{ margin: '0 0 1rem 0' }}>Default Probability</h3>
            <svg width={radius * 2} height={radius + strokeWidth} style={{ transform: 'rotate(180deg)' }}>
              {/* Background Arc */}
              <circle
                cx={radius} cy={radius} r={radius - strokeWidth/2}
                fill="none" stroke="var(--border-color)" strokeWidth={strokeWidth}
                strokeDasharray={dashArray} strokeDashoffset="0"
              />
              {/* Foreground Arc */}
              <circle
                cx={radius} cy={radius} r={radius - strokeWidth/2}
                fill="none" stroke={color} strokeWidth={strokeWidth}
                strokeDasharray={dashArray} strokeDashoffset={dashOffset}
                style={{ transition: 'stroke-dashoffset 0.5s ease-out, stroke 0.5s ease' }}
              />
            </svg>
            <div style={{ marginTop: '-40px', textAlign: 'center' }}>
              <div style={{ fontSize: '3rem', fontWeight: 800, color }}>{probability}%</div>
              <div style={{ color: 'var(--text-secondary)', fontWeight: 500, marginTop: '10px' }}>
                {probability > 60 ? 'High Risk' : probability > 30 ? 'Medium Risk' : 'Low Risk'}
              </div>
            </div>
          </div>

          {/* What-If Widget */}
          <div className="card" style={{ padding: '2rem' }}>
            <h3 style={{ margin: '0 0 1.5rem 0' }}>What-If Analysis</h3>
            
            <div style={{ marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <label style={{ fontWeight: 500 }}>Reported Income</label>
                <span style={{ color: 'var(--primary)', fontWeight: 600 }}>${income.toLocaleString()}</span>
              </div>
              <input 
                type="range" 
                min="20000" max="250000" step="5000"
                value={income} 
                onChange={e => setIncome(parseInt(e.target.value))}
                onMouseUp={() => fetchPrediction()}
                style={{ width: '100%' }}
              />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <label style={{ fontWeight: 500 }}>Credit Amount</label>
                <span style={{ color: 'var(--primary)', fontWeight: 600 }}>${creditAmount.toLocaleString()}</span>
              </div>
              <input 
                type="range" 
                min="10000" max="1000000" step="10000"
                value={creditAmount} 
                onChange={e => setCreditAmount(parseInt(e.target.value))}
                onMouseUp={() => fetchPrediction()}
                style={{ width: '100%' }}
              />
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          
          {/* SHAP Chart */}
          <div className="card" style={{ padding: '2rem', flex: 1 }}>
            <h3 style={{ margin: '0 0 1rem 0' }}>Feature Contributions (SHAP Values)</h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem', fontSize: '0.875rem' }}>
              Displays how specific variables drove the neural network's final decision. Red increases risk, green decreases it.
            </p>
            <div style={{ width: '100%', height: 300 }}>
              <ResponsiveContainer>
                <BarChart data={shapValues} layout="vertical" margin={{ left: 50 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="feature" width={100} />
                  <Tooltip formatter={(value: any) => Number(value).toFixed(3)} />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {shapValues.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.value > 0 ? 'var(--danger)' : 'var(--success)'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Explainability AI */}
          <div className="card" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ margin: 0 }}>AI Analyst Explanation</h3>
              <button className="btn-primary" onClick={handleGenerateExplanation} disabled={loadingExpl}>
                {loadingExpl ? 'Generating...' : 'Generate New Report'}
              </button>
            </div>
            
            <div style={{ 
              background: '#f8fafc', 
              padding: '1.5rem', 
              borderRadius: '0.5rem',
              border: '1px solid var(--border-color)',
              minHeight: '100px'
            }}>
              {explanation ? (
                <p style={{ lineHeight: 1.6, margin: 0 }}>{explanation}</p>
              ) : (
                <p style={{ color: 'var(--text-secondary)', margin: 0, fontStyle: 'italic', textAlign: 'center' }}>
                  Click "Generate New Report" to query GPT-4 for a plain-English translation of this applicant's deep-learning risk profile.
                </p>
              )}
            </div>
          </div>

        </div>

      </div>
    </>
  );
}
