import { useNavigate } from 'react-router-dom';

const mockCustomers = [
  { id: '100001', name: 'Maria Garcia', income: '$45,000', riskStatus: 'High Risk', score: 82 },
  { id: '100002', name: 'James Smith', income: '$120,000', riskStatus: 'Low Risk', score: 12 },
  { id: '100003', name: 'Sarah Connor', income: '$68,000', riskStatus: 'Medium Risk', score: 45 },
];

export default function CustomersPage() {
  const navigate = useNavigate();

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Customer Directory</h1>
        <div style={{ color: 'var(--text-secondary)' }}>Select a customer to view their ML risk profile</div>
      </div>
      
      <div className="card">
        <table className="table">
          <thead>
            <tr>
              <th>Application ID</th>
              <th>Applicant Name</th>
              <th>Reported Income</th>
              <th>Initial ML Risk</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {mockCustomers.map(c => (
              <tr key={c.id}>
                <td style={{ fontWeight: 500 }}>#{c.id}</td>
                <td>{c.name}</td>
                <td>{c.income}</td>
                <td>
                  <span className={`badge ${
                    c.score > 60 ? 'badge-danger' : c.score > 30 ? 'badge-warning' : 'badge-success'
                  }`}>
                    {c.riskStatus} ({c.score}%)
                  </span>
                </td>
                <td>
                  <button 
                    className="btn-primary" 
                    onClick={() => navigate(`/risk-profile?id=${c.id}&name=${encodeURIComponent(c.name)}`)}
                  >
                    Analyze Profile
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
