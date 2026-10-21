import React from 'react';
import { BookOpen, BarChart3, Database, SearchCode, Layers } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'search', label: 'Book Search', icon: BookOpen },
    { id: 'evaluation', label: 'Model Benchmark & Curves', icon: BarChart3 },
    { id: 'dataset', label: 'Dataset Analytics', icon: Database },
    { id: 'inspector', label: 'Inverted Index Inspector', icon: SearchCode },
  ];

  return (
    <header className="navbar">
      <div className="nav-brand">
        <div className="brand-icon">
          <Layers size={22} />
        </div>
        <div>
          <h1 className="brand-title">Book Search</h1>
          <p className="brand-subtitle">Information Retrieval — Book-Crossing Engine</p>
        </div>
      </div>

      <nav>
        <ul className="nav-links">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <li key={tab.id}>
                <button
                  className={`nav-tab ${isActive ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  <Icon size={16} />
                  <span>{tab.label}</span>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}
