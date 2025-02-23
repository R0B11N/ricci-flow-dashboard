import React, { useState, useEffect, useCallback } from 'react';
import './FutureRegimeAnalyzer.css';
import { useNavigate, useLocation } from 'react-router-dom';

function FutureRegimeAnalyzer() {
  const [selectedSectors, setSelectedSectors] = useState([]);
  const [timeframe, setTimeframe] = useState('7');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mounted, setMounted] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { tickers, startDate, endDate } = location.state || {};

  const sectors = [
    'Technology',
    'Financial',
    'Healthcare',
    'ConsumerCyclical',
    'Communication',
    'Industrial',
    'ConsumerDefensive',
    'Energy',
    'BasicMaterials',
    'RealEstate',
    'Utilities'
  ];

  // Display names mapping for UI
  const sectorDisplayNames = {
    'ConsumerCyclical': 'Consumer Cyclical',
    'ConsumerDefensive': 'Consumer Defensive',
    'BasicMaterials': 'Basic Materials',
    'RealEstate': 'Real Estate'
  };

  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);

  const toggleSector = useCallback((sector) => {
    if (!mounted) return;
    setSelectedSectors(prev => {
      const newSelection = prev.includes(sector) 
        ? prev.filter(s => s !== sector)
        : [...prev, sector];
      console.log("Selected sectors:", newSelection);  // Log selected sectors
      return newSelection;
    });
  }, [mounted]);

  const formatProbability = useCallback((value) => {
    if (value === undefined || value === null) return '0.0%';
    return `${value.toFixed(1)}%`;
  }, []);

  const formatSectorMovement = useCallback((value) => {
    if (value === undefined || value === null) return '0.00%';
    const formatted = (value * 100).toFixed(2);
    return `${formatted}%`;
  }, []);

  const handleTimeframeChange = useCallback((event) => {
    if (!mounted) return;
    setTimeframe(event.target.value);
    setPrediction(null);
  }, [mounted]);

  const analyzeFutureRegime = useCallback(async () => {
    if (!mounted || selectedSectors.length < 2) return;
    
    setLoading(true);
    setError(null);
    setPrediction(null);
    
    try {
      // Log what we're sending
      console.log("Sending sectors:", selectedSectors);
      
      const response = await fetch('http://localhost:8000/future-regime-analysis/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sectors: selectedSectors,  // Ensure this is an array of sector names 
          timeframe: Number(timeframe)
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze regime');
      }
      
      const data = await response.json();
      console.log("Received data:", data);  // Log what we receive
      
      if (mounted && data && data.regime_analysis) {
        setPrediction(data.regime_analysis);
      } else if (mounted) {
        throw new Error('Unexpected response format');
      }
    } catch (error) {
      if (mounted) {
        setError(error.message);
        console.error('Error:', error);
      }
    } finally {
      if (mounted) {
        setLoading(false);
      }
    }
  }, [mounted, selectedSectors, timeframe]);

  const formatMovement = (movement) => {
    if (!movement || typeof movement.magnitude === 'undefined') return '0.0%';
    
    const direction = movement.direction === 'up' ? '↑' : '↓';
    return `${direction}${movement.magnitude.toFixed(1)}%`;
  };

  const getMovementClass = (movement) => {
    if (!movement) return '';
    return movement.direction === 'up' ? 'positive' : 'negative';
  };

  const renderSectorMovements = (sectorMovements) => {
    if (!sectorMovements || Object.keys(sectorMovements).length === 0) {
      return <p>No sector movement predictions available</p>;
    }

    return (
      <div className="sector-movements">
        {Object.entries(sectorMovements).map(([sector, data]) => (
          <div key={sector} className="sector-movement-item">
            <div className="sector-name">
              {sectorDisplayNames[sector] || sector}
            </div>
            <div className={`movement-indicator ${data.direction}`}>
              {data.direction === 'up' ? 
                <span>↑</span> : 
                <span>↓</span>
              }
              <span className="magnitude">{data.magnitude.toFixed(1)}%</span>
            </div>
            <div className="confidence-bar">
              <div 
                className="confidence-level" 
                style={{width: `${data.confidence * 100}%`}}
              />
            </div>
          </div>
        ))}
      </div>
    );
  };

  // Add function to preserve state when going back (defunct? Save a fix for later)
  const handleBack = () => {
    navigate('/', { 
      state: { 
        preservedTickers: tickers,
        preservedStartDate: startDate,
        preservedEndDate: endDate
      } 
    });
  };

  if (!mounted) return null;

  return (
    <div className="analyzer-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <button
          className="back-button"
          onClick={handleBack}
        >
          Back to Analysis
        </button>
        <h1>Stock Market Regime Analysis</h1>
        <div style={{ width: '120px' }}></div>
      </div>
      
      {/* Show the current tickers for reference */}
      {tickers && (
        <div className="current-tickers">
          <p>Analyzing stocks: {tickers}</p>
        </div>
      )}
      
      <div className="input-card">
        <div className="sector-selection">
          <h3>Select Sectors</h3>
          <div className="sector-grid">
            {sectors.map(sector => (
              <div key={sector} className="sector-item">
                <label>
                  <input
                    type="checkbox"
                    checked={selectedSectors.includes(sector)}
                    onChange={() => toggleSector(sector)}
                  />
                  {sector}
                </label>
              </div>
            ))}
          </div>
        </div>

        <div className="timeframe-selection">
          <h3>Prediction Timeframe</h3>
          <div className="timeframe-options">
            {[7, 14, 30].map(days => (
              <button
                key={days}
                className={`timeframe-button ${timeframe === days ? 'active' : ''}`}
                onClick={() => setTimeframe(days)}
              >
                {days} Days
              </button>
            ))}
          </div>
        </div>

        <button 
          className="analyze-button"
          onClick={analyzeFutureRegime}
          disabled={selectedSectors.length === 0}
        >
          Analyze Future Regime
        </button>
      </div>

      {prediction && (
        <div className="results-card">
          <div className="regime-prediction">
            <h3>Predicted Regime Transition</h3>
            <div className="prediction-header">
              <span className="regime-label">Next Regime: {prediction.predicted_regime}</span>
              <div className="confidence-metrics">
                <span>Model Confidence: {(prediction.confidence_metrics?.model_confidence || 0).toFixed(1)}%</span>
                <span>Transition Probability: {(prediction.confidence_metrics?.transition_probability || 0).toFixed(1)}%</span>
              </div>
            </div>
          </div>

          <div className="regime-probabilities">
            <h3>Regime Probabilities</h3>
            <div className="probability-grid">
              {Object.entries(prediction.regime_probabilities || {}).map(([regime, prob]) => (
                <div key={regime} className="probability-item">
                  <span>{regime}</span>
                  <span>{(prob || 0).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          <div className="sector-predictions">
            <h3>Predicted Sector Movements</h3>
            <div className="movement-grid">
              {Object.entries(prediction.sector_movements || {}).map(([sector, movement]) => (
                <div key={sector} className="movement-item">
                  <span>{sector}</span>
                  <span className={getMovementClass(movement)}>
                    {formatMovement(movement)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {mounted && error && (
        <div className="error-message">
          <span>Error:</span> {error}
        </div>
      )}
    </div>
  );
}

export default React.memo(FutureRegimeAnalyzer); 