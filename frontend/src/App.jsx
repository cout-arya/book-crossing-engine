import React, { useState } from 'react';
import Navbar from './components/Navbar';
import SearchPage from './pages/SearchPage';
import EvaluationPage from './pages/EvaluationPage';
import DatasetPage from './pages/DatasetPage';
import IndexInspectorPage from './pages/IndexInspectorPage';

export default function App() {
  const [activeTab, setActiveTab] = useState('search');

  return (
    <div className="app-container">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="main-content">
        {activeTab === 'search' && <SearchPage />}
        {activeTab === 'evaluation' && <EvaluationPage />}
        {activeTab === 'dataset' && <DatasetPage />}
        {activeTab === 'inspector' && <IndexInspectorPage />}
      </main>

      <footer
        style={{
          borderTop: '1px solid var(--border-subtle)',
          padding: '1.5rem 2rem',
          textAlign: 'center',
          fontSize: '0.8rem',
          color: 'var(--text-dim)',
          background: 'var(--bg-secondary)',
          marginTop: 'auto',
        }}
      >
        <p>
          Book Search using Book-Crossing • Information Retrieval System •
          Algorithms: TF-IDF, BM25, Sentence Transformer (all-MiniLM-L6-v2), Hybrid RRF (k=60)
        </p>
      </footer>
    </div>
  );
}
