import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000/api';

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const searchBooks = async ({ query, model = 'hybrid', limit = 10, authorFilter = null, yearFilter = null }) => {
  const payload = {
    query,
    model,
    limit: Number(limit),
    author_filter: authorFilter ? String(authorFilter) : null,
    year_filter: yearFilter ? String(yearFilter) : null,
  };
  const response = await client.post('/search', payload);
  return response.data;
};

export const getHealth = async () => {
  const response = await client.get('/health');
  return response.data;
};

export const getDatasetStats = async () => {
  const response = await client.get('/dataset/stats');
  return response.data;
};

export const getEvaluationSummary = async () => {
  const response = await client.get('/evaluation/summary');
  return response.data;
};

export const getEvaluationComparison = async () => {
  const response = await client.get('/evaluation/comparison');
  return response.data;
};

export const getPrecisionRecallCurves = async () => {
  const response = await client.get('/evaluation/precision-recall');
  return response.data;
};

export const getEvaluationQueries = async () => {
  const response = await client.get('/evaluation/queries');
  return response.data;
};

export const getIndexingStats = async () => {
  const response = await client.get('/indexing/stats');
  return response.data;
};

export const inspectIndexQuery = async (query) => {
  const response = await client.get(`/indexing/inspect`, {
    params: { query },
  });
  return response.data;
};
