import PropTypes from 'prop-types';

const ScoreResult = ({ result }) => {
  const { score, confidence, details } = result;

  return (
    <section className="result" aria-live="polite" aria-label="Face score results">
      <div className="result__header">
        <div className="result__score">
          <span className="result__label">Face Score</span>
          <p className="result__value">{score.toFixed(2)}</p>
          <span className="result__confidence">Confidence: {(confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="result__summary">
          <h3>How your score was calculated</h3>
          <p>
            We blend the neural network prediction with geometry-based feature scores for a transparent, reproducible outcome.
            Confidence reflects the model probability returned by the FastAI classifier.
          </p>
        </div>
      </div>

      <dl className="result__grid">
        <div className="result__item">
          <dt>Deep Learning Score</dt>
          <dd>{details.deep_learning_score.toFixed(2)} / 5</dd>
        </div>
        <div className="result__item">
          <dt>Facial Feature Score</dt>
          <dd>{details.facial_feature_score.toFixed(2)} / 5</dd>
        </div>
        <div className="result__item">
          <dt>Eyes Score</dt>
          <dd>{details.eyes_score.toFixed(2)} / 5</dd>
        </div>
        <div className="result__item">
          <dt>Nose Score</dt>
          <dd>{details.nose_score.toFixed(2)} / 5</dd>
        </div>
      </dl>
    </section>
  );
};

ScoreResult.propTypes = {
  result: PropTypes.shape({
    score: PropTypes.number.isRequired,
    confidence: PropTypes.number.isRequired,
    details: PropTypes.shape({
      deep_learning_score: PropTypes.number.isRequired,
      facial_feature_score: PropTypes.number.isRequired,
      eyes_score: PropTypes.number.isRequired,
      nose_score: PropTypes.number.isRequired
    }).isRequired
  }).isRequired
};

export default ScoreResult;
