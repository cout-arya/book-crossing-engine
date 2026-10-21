import React, { useState } from 'react';
import { Book, Star, Calendar, Building, Hash } from 'lucide-react';

export default function BookCard({ book }) {
  const [imageFailed, setImageFailed] = useState(false);

  const hasCoverUrl = (book.image_url_m || book.image_url_l) && !imageFailed;
  const coverUrl = book.image_url_l || book.image_url_m;

  return (
    <div className="book-card">
      <div className="rank-badge">#{book.rank}</div>

      <div className="book-cover-container">
        {hasCoverUrl ? (
          <img
            src={coverUrl}
            alt={book.title}
            className="book-cover-img"
            loading="lazy"
            onError={() => setImageFailed(true)}
          />
        ) : (
          <div className="book-cover-placeholder">
            <Book size={28} />
            <span>No Cover</span>
          </div>
        )}
      </div>

      <div className="book-info">
        <h3 className="book-title" title={book.title}>
          {book.title}
        </h3>
        
        <p className="book-author">by {book.author || 'Unknown Author'}</p>

        <div className="book-metadata-row">
          <span title="Year of publication">
            <Calendar size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 2 }} />
            {book.year && book.year !== '0' ? book.year : 'Unknown Year'}
          </span>
          <span>•</span>
          <span title="Publisher">
            <Building size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 2 }} />
            {book.publisher || 'Unknown'}
          </span>
          <span>•</span>
          <span className="book-isbn" title="ISBN Identifier">
            <Hash size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 1 }} />
            {book.isbn}
          </span>
        </div>

        <div className="book-ratings-row">
          <span className="rating-badge" title="Average explicit user rating out of 10">
            <Star size={13} fill="currentColor" />
            {book.avg_rating > 0 ? book.avg_rating.toFixed(1) : 'Unrated'}
          </span>
          <span className="rating-count">
            ({book.rating_count} {book.rating_count === 1 ? 'rating' : 'ratings'})
          </span>
        </div>

        <div className="score-badge">
          <span className="score-text">
            Score: {typeof book.score === 'number' ? book.score.toFixed(4) : book.score}
          </span>

          {book.component_ranks && Object.keys(book.component_ranks).length > 0 && (
            <div className="component-ranks-row" title="Component model ranks before RRF fusion">
              {Object.entries(book.component_ranks).map(([model, r]) => (
                <span key={model} className="sub-rank-tag">
                  {model}:{r}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
