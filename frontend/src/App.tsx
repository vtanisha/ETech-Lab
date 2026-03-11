import { BrowserRouter as Router, Routes, Route, NavLink, Navigate } from 'react-router-dom';
import { Users, AlertTriangle, LayoutDashboard } from 'lucide-react';
import CustomersPage from './pages/CustomersPage';
import RiskProfilePage from './pages/RiskProfilePage';
import DashboardPage from './pages/DashboardPage';

function App() {
  return (
    <Router>
      <div className="app-container">
        <nav className="sidebar">
          <div style={{ marginBottom: '2rem', padding: '0 1rem', fontSize: '1.25rem', fontWeight: 800, color: 'var(--primary)' }}>
            CrediRisk
          </div>
          
          <NavLink to="/customers" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <Users size={20} /> Customer Directory
          </NavLink>
          
          <NavLink to="/risk-profile" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <AlertTriangle size={20} /> ML Risk Profile
          </NavLink>

          <NavLink to="/dashboard" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <LayoutDashboard size={20} /> Analytics
          </NavLink>
        </nav>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<Navigate to="/customers" replace />} />
            <Route path="/customers" element={<CustomersPage />} />
            <Route path="/risk-profile" element={<RiskProfilePage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
