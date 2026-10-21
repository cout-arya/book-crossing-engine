import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Database, Users, Star, BookMarked, Filter, Layers } from 'lucide-react';
import { getDatasetStats } from '../services/api';

// Decode HTML entities (e.g. &amp; → &) that may come from the backend CSV parser
function decodeHtml(str) {
  const txt = document.createElement('textarea');
  txt.innerHTML = str;
  return txt.value;
}

export default function DatasetPage() {
  const [stats, setStats] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadStats() {
      try {
        setIsLoading(true);
        const data = await getDatasetStats();
        setStats(data);
      } catch (err) {
        console.error('Failed to load dataset stats:', err);
        setError('Could not load dataset analytics. Ensure preprocessing script has completed.');
      } finally {
        setIsLoading(false);
      }
    }
    loadStats();
  }, []);

  if (isLoading) {
    return (
      <div className="state-center">
        <div className="spinner"></div>
        <p>Loading Book-Crossing dataset statistics...</p>
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="state-center">
        <p style={{ color: 'var(--accent-rose)' }}>{error}</p>
      </div>
    );
  }

  // Format year distribution for Recharts
  const yearData = Object.entries(stats.year_distribution || {}).map(([era, count]) => ({
    era,
    count,
  }));

  // Format rating distribution (filter ratings 1-10 vs 0)
  const ratingData = Object.entries(stats.rating_distribution || {})
    .filter(([rating]) => Number(rating) > 0)
    .map(([rating, count]) => ({
      rating: `Star ${rating}`,
      count,
    }));

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Book-Crossing Dataset Analytics</h2>
        <p className="page-description">
          Detailed metrics from the original Book-Crossing corpus containing ~271k books,
          ~1.15M rating records, and ~278k user demographic profiles.
        </p>
      </div>

      {/* Metric Cards Grid */}
      <div className="stats-cards-grid">
        <div className="metric-card">
          <span className="metric-label">Books Retained</span>
          <span className="metric-val">{(stats.books_retained || 0).toLocaleString()}</span>
          <span className="metric-sub">From {(stats.books_loaded || 0).toLocaleString()} loaded</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Total User Ratings</span>
          <span className="metric-val">{(stats.ratings_loaded || 0).toLocaleString()}</span>
          <span className="metric-sub">{(stats.explicit_ratings_count || 0).toLocaleString()} explicit (&gt;0)</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Implicit Ratings (=0)</span>
          <span className="metric-val">{(stats.implicit_ratings_count || 0).toLocaleString()}</span>
          <span className="metric-sub">Logged interactions / reads</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Registered Users</span>
          <span className="metric-val">{(stats.users_loaded || 0).toLocaleString()}</span>
          <span className="metric-sub">Demographic user records</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Duplicates Removed</span>
          <span className="metric-val">{(stats.duplicates_removed || 0).toLocaleString()}</span>
          <span className="metric-sub">Deduplicated by ISBN</span>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="charts-grid">
        {/* Publication Year Chart */}
        <div className="chart-card">
          <h4 className="chart-title">Publication Year Distribution</h4>
          <p className="chart-subtitle">Breakdown of catalog titles across publication eras</p>
          <div style={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
              <BarChart data={yearData} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="era" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#fff' }}
                />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Books" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Explicit Rating Distribution */}
        <div className="chart-card">
          <h4 className="chart-title">Explicit User Rating Distribution (1–10)</h4>
          <p className="chart-subtitle">Distribution of 433,671 explicit ratings (excluding 0-implicit)</p>
          <div style={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
              <BarChart data={ratingData} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="rating" stroke="#9ca3af" tick={{ fontSize: 10 }} />
                <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#fff' }}
                />
                <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} name="Ratings" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Top Authors and Publishers Tables */}
      <div className="charts-grid">
        <div className="table-container">
          <table className="custom-table">
            <thead>
              <tr>
                <th>Top 10 Authors</th>
                <th style={{ textAlign: 'right' }}>Title Count</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(stats.top_authors || {}).map(([author, count]) => (
                <tr key={author}>
                  <td style={{ fontWeight: 600 }}>{author}</td>
                  <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                    {count.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="table-container">
          <table className="custom-table">
            <thead>
              <tr>
                <th>Top 10 Publishers</th>
                <th style={{ textAlign: 'right' }}>Title Count</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(stats.top_publishers || {}).map(([publisher, count]) => (
                <tr key={publisher}>
                  <td style={{ fontWeight: 600 }}>{decodeHtml(publisher)}</td>
                  <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                    {count.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
