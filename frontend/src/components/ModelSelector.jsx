import React from 'react';

const MODELS = [
  {
    id: 'hybrid',
    name: 'Hybrid RRF (Recommended)',
    desc: 'Fuses TF-IDF, BM25 & Dense Semantic with Reciprocal Rank Fusion (k=60)',
  },
  {
    id: 'bm25',
    name: 'BM25',
    desc: 'Probabilistic term frequency saturation and document length normalization (k1=1.5, b=0.75)',
  },
  {
    id: 'tfidf',
    name: 'TF-IDF',
    desc: 'Term Frequency - Inverse Document Frequency with sublinear term scaling',
  },
  {
    id: 'semantic',
    name: 'Sentence Transformer',
    desc: 'Dense semantic embeddings via all-MiniLM-L6-v2 deep transformer',
  },
];

export default function ModelSelector({ selectedModel, onSelectModel, modelAvailability = {} }) {
  return (
    <div className="control-item">
      <label className="control-label">Retrieval Ranking Algorithm</label>
      <select
        className="control-select"
        value={selectedModel}
        onChange={(e) => onSelectModel(e.target.value)}
      >
        {MODELS.map((m) => {
          const isAvail = modelAvailability[m.id] !== false;
          return (
            <option key={m.id} value={m.id}>
              {m.name} {!isAvail ? '(Unavailable)' : ''}
            </option>
          );
        })}
      </select>
    </div>
  );
}
