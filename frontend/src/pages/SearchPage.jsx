import React, { useState, useEffect, useCallback } from 'react';
import { Search, Sparkles, Filter, AlertCircle, Clock, CheckCircle } from 'lucide-react';
import { searchBooks } from '../services/api';
import BookCard from '../components/BookCard';
import ModelSelector from '../components/ModelSelector';

const SUGGESTED_QUERIES = [
  'The Da Vinci Code',
  'To Kill a Mockingbird',
  'Stephen King',
  'Harry Potter',
  'Agatha Christie',
  'murder mystery detective',
  '1984',
  'space science fiction',
  'magic wizard fantasy',
];

export default function SearchPage() {
  const [query, setQuery] = useState('The Da Vinci Code');
  const [selectedModel, setSelectedModel] = useState('hybrid');
  const [limit, setLimit] = useState(12);
  const [authorFilter, setAuthorFilter] = useState('');
  const [yearFilter, setYearFilter] = useState('');

  const [results, setResults] = useState([]);
  const [totalResults, setTotalResults] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [modelAvailability, setModelAvailability] = useState({});
  const [statusMessage, setStatusMessage] = useState(null);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = useCallback(async (overrideQuery = null) => {
    const q = overrideQuery !== null ? overrideQuery : query;
    if (!q.trim()) return;

    setIsLoading(true);
    setError(null);
    setStatusMessage(null);

    try {
      const data = await searchBooks({
        query: q,
        model: selectedModel,
        limit,
        authorFilter: authorFilter.trim() || null,
        yearFilter: yearFilter.trim() || null,
      });

      setResults(data.results || []);
      setTotalResults(data.total_results || 0);
      setLatencyMs(data.latency_ms || 0);
      setModelAvailability(data.model_availability || {});
      if (data.message) {
        setStatusMessage(data.message);
      }
      setHasSearched(true);
    } catch (err) {
      console.error('Search error:', err);
      setError(
        err.response?.data?.detail ||
        'Failed to execute search. Make sure the backend API is running.'
      );
    } finally {
      setIsLoading(false);
    }
  }, [query, selectedModel, limit, authorFilter, yearFilter]);

  // Initial search on mount
  useEffect(() => {
    handleSearch('The Da Vinci Code');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSuggestedClick = (suggested) => {
    setQuery(suggested);
    handleSearch(suggested);
  };

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Book Search Engine</h2>
        <p className="page-description">
          Search over 271,000 Book-Crossing titles, authors, and metadata using state-of-the-art
          lexical, dense semantic, and hybrid reciprocal rank fusion algorithms.
        </p>
      </div>

      <section className="search-section">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSearch();
          }}
        >
          <div className="search-input-group">
            <div className="search-input-wrapper">
              <Search className="search-icon-inside" size={20} />
              <input
                type="text"
                className="search-input"
                placeholder="Search by book title, author, genre, or keyword..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            <button type="submit" className="btn-search" disabled={isLoading}>
              {isLoading ? 'Searching...' : 'Search'}
            </button>
          </div>

          <div className="suggested-queries">
            <span className="suggested-label">Try:</span>
            {SUGGESTED_QUERIES.map((sq) => (
              <button
                type="button"
                key={sq}
                className="query-pill"
                onClick={() => handleSuggestedClick(sq)}
              >
                {sq}
              </button>
            ))}
          </div>

          <div className="controls-grid">
            <ModelSelector
              selectedModel={selectedModel}
              onSelectModel={(m) => setSelectedModel(m)}
              modelAvailability={modelAvailability}
            />

            <div className="control-item">
              <label className="control-label">Author Filter (Optional)</label>
              <input
                type="text"
                className="control-input"
                placeholder="e.g. Dan Brown, King"
                value={authorFilter}
                onChange={(e) => setAuthorFilter(e.target.value)}
              />
            </div>

            <div className="control-item">
              <label className="control-label">Publication Year (Optional)</label>
              <input
                type="text"
                className="control-input"
                placeholder="e.g. 2003, 1990"
                value={yearFilter}
                onChange={(e) => setYearFilter(e.target.value)}
              />
            </div>

            <div className="control-item">
              <label className="control-label">Result Limit</label>
              <select
                className="control-select"
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value))}
              >
                <option value={6}>Top 6</option>
                <option value={10}>Top 10 (Assignment standard)</option>
                <option value={12}>Top 12</option>
                <option value={20}>Top 20</option>
                <option value={50}>Top 50</option>
              </select>
            </div>
          </div>
        </form>
      </section>

      {statusMessage && (
        <div className="note-box" style={{ borderColor: 'var(--accent-amber)' }}>
          <strong>Notice: </strong>
          {statusMessage}
        </div>
      )}

      {error && (
        <div className="note-box" style={{ borderColor: 'var(--accent-rose)' }}>
          <strong>Search Error: </strong>
          {error}
        </div>
      )}

      {/* Results Metadata Bar */}
      {hasSearched && !isLoading && (
        <div className="results-meta-bar">
          <div className="results-count">
            Found <strong>{totalResults}</strong> ranked books for "<em>{query}</em>"
          </div>
          <div className="results-stats-pills">
            <span className="meta-pill">
              <Clock size={12} style={{ display: 'inline', marginRight: 4 }} />
              {latencyMs} ms
            </span>
            <span className="meta-pill">Model: {selectedModel.toUpperCase()}</span>
          </div>
        </div>
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="state-center">
          <div className="spinner"></div>
          <p>Running retrieval algorithms across 271,045 catalog records...</p>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && hasSearched && results.length === 0 && (
        <div className="state-center">
          <AlertCircle size={40} className="state-icon" />
          <h3>No matching books found</h3>
          <p>Try broadening your query or removing author/year filters.</p>
        </div>
      )}

      {/* Book Grid */}
      {!isLoading && results.length > 0 && (
        <div className="book-grid">
          {results.map((book) => (
            <BookCard key={book.isbn + '-' + book.rank} book={book} />
          ))}
        </div>
      )}
    </div>
  );
}
