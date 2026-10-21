import React, { useState, useEffect } from 'react';
import { SearchCode, Hash, FileCode, CheckCircle, Search } from 'lucide-react';
import { getIndexingStats, inspectIndexQuery } from '../services/api';

export default function IndexInspectorPage() {
  const [indexStats, setIndexStats] = useState(null);
  const [query, setQuery] = useState('harry potter rowling');
  const [inspectionResult, setInspectionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchStats() {
      try {
        const stats = await getIndexingStats();
        setIndexStats(stats);
      } catch (err) {
        console.error('Failed to load indexing stats:', err);
      }
    }
    fetchStats();
    handleInspect('harry potter rowling');
  }, []);

  const handleInspect = async (qToInspect) => {
    const q = qToInspect || query;
    if (!q.trim()) return;

    setIsLoading(true);
    setError(null);
    try {
      const data = await inspectIndexQuery(q);
      setInspectionResult(data);
    } catch (err) {
      console.error('Inspection failed:', err);
      setError('Failed to inspect inverted index. Make sure the backend index is loaded.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Metadata Inverted Index Inspector</h2>
        <p className="page-description">
          Technical inspection interface demonstrating the inner mechanics of inverted indexing:
          tokenization, Porter stemming, vocabulary mapping, postings lists, and document frequencies.
        </p>
      </div>

      {/* Index Structural Stats */}
      {indexStats && (
        <div className="stats-cards-grid">
          <div className="metric-card">
            <span className="metric-label">Indexed Catalog Books</span>
            <span className="metric-val">{(indexStats.total_documents || 0).toLocaleString()}</span>
            <span className="metric-sub">Total documents (N)</span>
          </div>

          <div className="metric-card">
            <span className="metric-label">Inverted Vocabulary</span>
            <span className="metric-val">{(indexStats.vocabulary_size || 0).toLocaleString()}</span>
            <span className="metric-sub">Unique stemmed terms</span>
          </div>

          <div className="metric-card">
            <span className="metric-label">Postings Entries</span>
            <span className="metric-val">{(indexStats.total_postings_entries || 0).toLocaleString()}</span>
            <span className="metric-sub">DocID & TF tuples</span>
          </div>

          <div className="metric-card">
            <span className="metric-label">Avg Document Length</span>
            <span className="metric-val">{indexStats.avg_document_length || 0}</span>
            <span className="metric-sub">Tokens per document</span>
          </div>
        </div>
      )}

      {/* Interactive Query Inspector */}
      <section className="search-section">
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.75rem', color: '#fff' }}>
          Inspect Query Postings
        </h3>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleInspect();
          }}
        >
          <div className="search-input-group">
            <div className="search-input-wrapper">
              <SearchCode className="search-icon-inside" size={20} />
              <input
                type="text"
                className="search-input"
                placeholder="Enter query terms to inspect postings (e.g. da vinci, mockingbird)..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            <button type="submit" className="btn-search" disabled={isLoading}>
              {isLoading ? 'Inspecting...' : 'Inspect Index'}
            </button>
          </div>
        </form>
      </section>

      {error && (
        <div className="note-box" style={{ borderColor: 'var(--accent-rose)' }}>
          <strong>Error: </strong>
          {error}
        </div>
      )}

      {inspectionResult && (
        <div>
          <div className="note-box">
            <strong>Lexical Pipeline Output: </strong>
            Query "<em>{inspectionResult.query}</em>" was preprocessed into normalized stemmed tokens:
            <div style={{ marginTop: '0.5rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {inspectionResult.processed_tokens.map((token, idx) => (
                <span
                  key={idx}
                  style={{
                    background: 'var(--bg-input)',
                    border: '1px solid var(--accent-primary)',
                    color: 'var(--accent-cyan)',
                    padding: '0.2rem 0.6rem',
                    borderRadius: 4,
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.85rem',
                  }}
                >
                  {token}
                </span>
              ))}
            </div>
          </div>

          <div className="results-meta-bar">
            <div className="results-count">
              Total matched candidate documents in Inverted Index:{' '}
              <strong style={{ color: 'var(--accent-emerald)' }}>
                {inspectionResult.total_candidate_documents.toLocaleString()}
              </strong>
            </div>
          </div>

          {/* Postings Breakdown Table */}
          <div className="table-container">
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Stemmed Index Term</th>
                  <th>Document Frequency (DF)</th>
                  <th>Matching Books Count</th>
                  <th>Sample Candidate ISBNs</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(inspectionResult.terms_matched || {}).map(([term, info]) => (
                  <tr key={term}>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-indigo)' }}>
                      {term}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{info.term_doc_frequency.toLocaleString()}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{info.matched_doc_count.toLocaleString()}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      {info.sample_isbns.join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
