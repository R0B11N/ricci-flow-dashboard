import React, { useState, useCallback, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { Line } from 'react-plotly.js';

const styles = {
  '@keyframes gradient': {
    '0%': { backgroundPosition: '0% 50%' },
    '50%': { backgroundPosition: '100% 50%' },
    '100%': { backgroundPosition: '0% 50%' }
  },
  container: {
    padding: '20px',
    maxWidth: '1000px',
    margin: '0 auto',
    fontFamily: 'Arial, sans-serif',
    minHeight: '100vh',
    boxSizing: 'border-box',
    position: 'relative',
    zIndex: 1,
  },
  background: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab)',
    backgroundSize: '400% 400%',
    animation: 'gradient 15s ease infinite',
    zIndex: 0,
    opacity: 0.15
  },
  title: {
    color: '#2E3B4E',
    textAlign: 'center',
    marginBottom: '30px',
    fontSize: '2.5em',
    textShadow: '0 0 20px rgba(255,255,255,0.8)',
  },
  card: {
    background: 'rgba(255, 255, 255, 0.9)',
    padding: '30px',
    borderRadius: '15px',
    boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.15)',
    backdropFilter: 'blur(4px)',
    border: '1px solid rgba(255, 255, 255, 0.18)',
    marginBottom: '30px',
    transition: 'transform 0.3s ease, box-shadow 0.3s ease',
    ':hover': {
      transform: 'translateY(-5px)',
      boxShadow: '0 12px 40px 0 rgba(31, 38, 135, 0.25)',
    }
  },
  input: {
    width: '100%',
    padding: '12px',
    marginTop: '8px',
    borderRadius: '8px',
    border: '1px solid rgba(46, 59, 78, 0.2)',
    fontSize: '16px',
    transition: 'all 0.3s ease',
    background: 'rgba(255, 255, 255, 0.9)',
    ':focus': {
      boxShadow: '0 0 15px rgba(66, 133, 244, 0.3)',
      border: '1px solid #4285f4',
      outline: 'none'
    }
  },
  button: {
    padding: '12px 24px',
    backgroundColor: '#4285f4',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '16px',
    transition: 'all 0.3s ease',
    boxShadow: '0 4px 15px rgba(66, 133, 244, 0.3)',
    ':hover': {
      backgroundColor: '#3367d6',
      transform: 'translateY(-2px)',
      boxShadow: '0 6px 20px rgba(66, 133, 244, 0.4)',
    }
  },
  buttonContainer: {
    display: 'flex',
    gap: '10px',
    marginTop: '20px'
  },
  analysisContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
    padding: '20px',
    maxWidth: '1200px',
    margin: '0 auto'
  },
  analysisSection: {
    background: 'rgba(255, 255, 255, 0.95)',
    borderRadius: '15px',
    padding: '20px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
  },
  sectionTitle: {
    color: '#2E3B4E',
    marginBottom: '15px',
    borderBottom: '2px solid #4285f4',
    paddingBottom: '5px'
  },
  anomalyCard: {
    padding: '15px',
    marginBottom: '10px',
    borderRadius: '8px',
    border: '2px solid',
    background: 'rgba(255, 255, 255, 0.9)'
  },
  pairCard: {
    padding: '15px',
    marginBottom: '10px',
    borderRadius: '8px',
    border: '2px solid #4285f4',
    background: 'rgba(255, 255, 255, 0.9)'
  },
  pairMetrics: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
    gap: '10px',
    margin: '10px 0'
  },
  metric: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '5px',
    backgroundColor: 'rgba(66, 133, 244, 0.1)',
    borderRadius: '4px'
  },
  sectorInfo: {
    marginTop: '10px',
    padding: '5px',
    backgroundColor: 'rgba(52, 168, 83, 0.1)',
    borderRadius: '4px'
  },
  explanationCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    padding: '15px',
    borderRadius: '8px',
    marginBottom: '20px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  },
  insightsList: {
    listStyle: 'none',
    padding: '0',
    margin: '10px 0',
    '& li': {
      marginBottom: '8px',
      lineHeight: '1.4'
    }
  },
  keyInsights: {
    backgroundColor: 'rgba(66, 133, 244, 0.1)',
    padding: '15px',
    borderRadius: '8px',
    marginTop: '15px'
  },
  tradingImplications: {
    backgroundColor: 'rgba(52, 168, 83, 0.1)',
    padding: '15px',
    borderRadius: '8px',
    marginTop: '15px'
  },
  regimeStats: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '20px',
    marginTop: '20px'
  },
  regimeCard: {
    padding: '20px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  },
  regimeMetrics: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '10px',
    marginTop: '10px'
  },
  noPairsCard: {
    padding: '20px',
    backgroundColor: 'rgba(251, 188, 4, 0.1)',
    borderRadius: '8px',
    marginTop: '20px'
  },
  leadersGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '20px',
    marginTop: '20px'
  },
  leaderCard: {
    padding: '15px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    backgroundColor: 'rgba(66,133,244,0.1)'
  },
  relationshipsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '20px',
    marginTop: '20px'
  },
  relationshipCard: {
    padding: '15px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    backgroundColor: 'white'
  },
  relationshipHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '10px'
  },
  arrow: {
    color: '#4285f4',
    fontSize: '20px'
  },
  subsectionTitle: {
    color: '#2E3B4E',
    marginBottom: '15px'
  },
  metric: {
    display: 'flex',
    justifyContent: 'space-between',
    margin: '5px 0'
  },
  predictionsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '20px',
    marginTop: '20px'
  },
  predictionCard: {
    padding: '15px',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    backgroundColor: 'white',
    transition: 'transform 0.2s',
    '&:hover': {
      transform: 'translateY(-2px)'
    }
  },
  pairTitle: {
    margin: '0 0 10px 0',
    color: '#1a73e8'
  },
  predictionMetrics: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px'
  },
  plotContainer: {
    marginBottom: '30px',
    padding: '20px',
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  },
  bestPairsContainer: {
    marginBottom: '30px'
  },
  bestPairsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '20px'
  },
  bestPairCard: {
    padding: '20px',
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
  },
  metricGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr',
    gap: '10px'
  }
};

function App() {
  // Get today's date in YYYY-MM-DD format
  const today = new Date().toISOString().split('T')[0];
  
  // Initialize state with today as end date
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState(today);
  const [tickers, setTickers] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [selectedNode, setSelectedNode] = useState(null);
  const [timeWindow, setTimeWindow] = useState(30); // Days for rolling window
  const [minCurvature, setMinCurvature] = useState(-1);
  const [maxCurvature, setMaxCurvature] = useState(1);
  const [minMarketCap, setMinMarketCap] = useState(0);
  const [selectedSectors, setSelectedSectors] = useState([]);
  const [volumeThreshold, setVolumeThreshold] = useState(0);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [showPredictions, setShowPredictions] = useState(false);
  const [regimeData, setRegimeData] = useState(null);
  const [showRegimes, setShowRegimes] = useState(false);
  const [leadLagData, setLeadLagData] = useState(null);
  const [predictions, setPredictions] = useState(null);

  // Sample sectors - we'll fetch these from the backend later (remind me)
  const sectors = [
    'Technology',
    'Financial Services',
    'Healthcare',
    'Consumer Cyclical',
    'Communication Services',
    'Industrial',
    'Consumer Defensive',
    'Energy',
    'Basic Materials',
    'Real Estate',
    'Utilities'
  ];

  // Fetch additional stock data for selected node
  const fetchStockDetails = useCallback(async (ticker) => {
    try {
      const response = await axios.get(`http://localhost:8000/stock-details/`, {
        params: { ticker }
      });
      return response.data;
    } catch (err) {
      console.error('Error fetching stock details:', err);
      return null;
    }
  }, []);

  const handleNodeClick = async (ticker) => {
    setSelectedNode(ticker);
    const details = await fetchStockDetails(ticker);
    // Update UI with details...
  };

  // Reset analysis when tickers or result changes
  useEffect(() => {
    setAnalysisResult(null);
    setShowPredictions(false);
  }, [tickers, result]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setAnalysisResult(null); // Reset analysis when new calculation starts
    setShowPredictions(false);
    
    try {
        // Validate inputs
        if (!tickers.trim()) {
            throw new Error('Please enter at least two stock tickers');
        }

        // Clean and format tickers
        const tickerList = tickers
            .split(',')
            .map(t => t.trim().toUpperCase())
            .filter(t => t)
            .join(',');

        if (tickerList.split(',').length < 2) {
            throw new Error('Please enter at least two stock tickers');
        }

        // Create URL with query parameters
        const url = new URL('http://localhost:8000/ricci-curvature/');
        url.searchParams.append('tickers', tickerList);
        url.searchParams.append('start', startDate);
        url.searchParams.append('end', endDate);

        console.log('Making request to:', url.toString());

        const response = await axios.get(url.toString());

        console.log('Response:', response.data);

        if (response.data && response.data.status === 'success') {
            setResult(response.data);
        } else {
            throw new Error('Invalid response from server');
        }
    } catch (err) {
        console.error('API Error:', err);
        setError(
            err.response?.data?.detail || 
            err.message || 
            'An error occurred while fetching data'
        );
    } finally {
        setLoading(false);
    }
  };

  // Add these functions near the top with other handlers
  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Update the handler functions
  const handleAnalyze = async () => {
    if (!result) {
      setError('Please calculate network first');
      return;
    }

    try {
      const response = await axios.get('http://localhost:8000/market-analysis/', {
        params: {
          tickers: tickers,
          start: startDate,
          end: endDate
        }
      });
      setAnalysisResult(response.data);
      // Add scroll after setting the result
      setTimeout(() => scrollToSection('analysisSection'), 100);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze market');
    }
  };

  const handleRegimeAnalysis = async () => {
    if (!result) {
      setError('Please calculate network first');
      return;
    }

    try {
      const response = await axios.get('http://localhost:8000/regime-analysis/', {
        params: {
          tickers: tickers,
          start: startDate,
          end: endDate
        }
      });
      setRegimeData(response.data);
      setShowRegimes(true);
      // Add scroll after setting the result
      setTimeout(() => scrollToSection('regimeSection'), 100);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze market regimes');
    }
  };

  const handleLeadLagAnalysis = async () => {
    if (!result) {
      setError('Please calculate network first');
      return;
    }

    try {
      const response = await axios.get('http://localhost:8000/lead-lag-analysis/', {
        params: {
          tickers: tickers,
          start: startDate,
          end: endDate
        }
      });
      setLeadLagData(response.data);
      setTimeout(() => scrollToSection('leadLagSection'), 100);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to analyze lead-lag relationships');
    }
  };

  const handlePredictions = async () => {
    try {
      const response = await axios.get('http://localhost:8000/predictions/', {
        params: {
          tickers: tickers,
          start: startDate,
          end: endDate
        }
      });
      setPredictions(response.data);
      setShowPredictions(true);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate predictions');
    }
  };

  const renderFilters = () => (
    <div style={{
      background: 'rgba(255, 255, 255, 0.9)',
      padding: '20px',
      borderRadius: '10px',
      marginBottom: '20px',
      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
    }}>
      <h3 style={{ marginBottom: '15px', color: '#2E3B4E' }}>Filters</h3>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {/* Curvature Range Filter */}
        <div>
          <label style={{ display: 'block', marginBottom: '10px', color: '#2E3B4E' }}>
            Curvature Range:
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <input
                type="range"
                min="-1"
                max="1"
                step="0.1"
                value={minCurvature}
                onChange={(e) => setMinCurvature(parseFloat(e.target.value))}
                style={{ flex: 1 }}
              />
              <span style={{ minWidth: '60px', textAlign: 'right' }}>
                {minCurvature.toFixed(2)}
              </span>
            </div>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <input
                type="range"
                min="-1"
                max="1"
                step="0.1"
                value={maxCurvature}
                onChange={(e) => setMaxCurvature(parseFloat(e.target.value))}
                style={{ flex: 1 }}
              />
              <span style={{ minWidth: '60px', textAlign: 'right' }}>
                {maxCurvature.toFixed(2)}
              </span>
            </div>
          </label>
        </div>

        {/* Market Cap Filter */}
        <div>
          <label style={{ display: 'block', marginBottom: '10px', color: '#2E3B4E' }}>
            Min Market Cap (Billions):
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <input
                type="range"
                min="0"
                max="1000"
                step="10"
                value={minMarketCap}
                onChange={(e) => setMinMarketCap(parseInt(e.target.value))}
                style={{ flex: 1 }}
              />
              <span style={{ minWidth: '60px', textAlign: 'right' }}>
                ${minMarketCap}B
              </span>
            </div>
          </label>
        </div>

        {/* Volume Filter */}
        <div>
          <label style={{ display: 'block', marginBottom: '10px', color: '#2E3B4E' }}>
            Min Daily Volume (Millions):
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <input
                type="range"
                min="0"
                max="1000"
                step="10"
                value={volumeThreshold}
                onChange={(e) => setVolumeThreshold(parseInt(e.target.value))}
                style={{ flex: 1 }}
              />
              <span style={{ minWidth: '60px', textAlign: 'right' }}>
                {volumeThreshold}M
              </span>
            </div>
          </label>
        </div>

        {/* Sector Filter */}
        <div>
          <label style={{ display: 'block', marginBottom: '10px', color: '#2E3B4E' }}>
            Sectors:
            <div style={{ 
              display: 'flex', 
              flexWrap: 'wrap', 
              gap: '5px', 
              maxHeight: '100px', 
              overflowY: 'auto' 
            }}>
              {sectors.map(sector => (
                <label key={sector} style={{ 
                  display: 'inline-flex',
                  alignItems: 'center',
                  padding: '4px 8px',
                  backgroundColor: selectedSectors.includes(sector) 
                    ? 'rgba(66,133,244,0.1)' 
                    : 'transparent',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  border: '1px solid #ddd',
                  fontSize: '14px'
                }}>
                  <input
                    type="checkbox"
                    checked={selectedSectors.includes(sector)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedSectors([...selectedSectors, sector]);
                      } else {
                        setSelectedSectors(selectedSectors.filter(s => s !== sector));
                      }
                    }}
                    style={{ marginRight: '5px' }}
                  />
                  {sector}
                </label>
              ))}
            </div>
          </label>
        </div>
      </div>

      {/* Reset Filters Button */}
      <button
        onClick={() => {
          setMinCurvature(-1);
          setMaxCurvature(1);
          setMinMarketCap(0);
          setSelectedSectors([]);
          setVolumeThreshold(0);
        }}
        style={{
          marginTop: '15px',
          padding: '8px 16px',
          backgroundColor: '#4285f4',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '14px'
        }}
      >
        Reset Filters
      </button>
    </div>
  );

  const renderGraph = () => {
    // Early return if no result or missing metadata
    if (!result || !result.metadata || !result.curvature) return null;

    // Filter nodes based on market cap and volume
    const filteredNodes = [...new Set(
        Object.keys(result.curvature).flatMap(key => key.split('-'))
    )].filter(node => {
        const marketCapInBillions = (result.metadata.market_caps[node] || 0) / 1e9;
        const volumeInMillions = (result.metadata.volumes[node] || 0) / 1e6;
        const sector = result.metadata.sectors[node];
        
        // Apply filters
        const passesMarketCap = marketCapInBillions >= minMarketCap;
        const passesVolume = volumeInMillions >= volumeThreshold;
        const passesSector = selectedSectors.length === 0 || selectedSectors.includes(sector);
        
        return passesMarketCap && passesVolume && passesSector;
    });

    // Filter edges based on curvature range and filtered nodes
    const filteredCurvature = Object.entries(result.curvature)
        .filter(([key, value]) => {
            const [node1, node2] = key.split('-');
            return (
                filteredNodes.includes(node1) &&
                filteredNodes.includes(node2) &&
                value >= minCurvature &&
                value <= maxCurvature
            );
        })
        .reduce((acc, [key, value]) => {
            acc[key] = value;
            return acc;
        }, {});

    // Calculate node positions for filtered nodes
    const positions = calculateForceDirectedLayout(filteredNodes, filteredCurvature);

    // Format helpers
    const formatMarketCap = (cap) => {
        if (cap >= 1e12) return `$${(cap / 1e12).toFixed(2)}T`;
        if (cap >= 1e9) return `$${(cap / 1e9).toFixed(2)}B`;
        if (cap >= 1e6) return `$${(cap / 1e6).toFixed(2)}M`;
        return `$${cap.toFixed(2)}`;
    };

    const formatVolume = (vol) => {
        if (vol >= 1e9) return `${(vol / 1e9).toFixed(2)}B`;
        if (vol >= 1e6) return `${(vol / 1e6).toFixed(2)}M`;
        return vol.toFixed(2);
    };

    // Node trace with filtered data
    const nodeTrace = {
        x: filteredNodes.map(node => positions[node].x),
        y: filteredNodes.map(node => positions[node].y),
        mode: 'markers+text',
        type: 'scatter',
        name: 'Stocks',
        text: filteredNodes,
        textposition: 'top center',
        marker: {
            size: filteredNodes.map(node => {
                const marketCap = result.metadata.market_caps[node] || 0;
                return Math.max(20, Math.min(50, Math.log10(marketCap/1e8)));
            }),
            color: filteredNodes.map(node => result.metadata.volatilities[node] || 0),
            colorscale: 'Viridis',
            showscale: true,
            colorbar: {
                title: 'Volatility',
                thickness: 15,
                len: 0.5,
                y: 0.8
            }
        },
        hoverinfo: 'text',
        hovertext: filteredNodes.map(node => `
            <b>${node}</b><br>
            Market Cap: ${formatMarketCap(result.metadata.market_caps[node] || 0)}<br>
            Daily Volume: ${formatVolume(result.metadata.volumes[node] || 0)}<br>
            Volatility: ${((result.metadata.volatilities[node] || 0) * 100).toFixed(2)}%<br>
            Sector: ${result.metadata.sectors[node] || 'Unknown'}
        `)
    };

    // Edge traces with filtered data
    const edgeTraces = Object.entries(filteredCurvature)
        .map(([key, value]) => {
            const [source, target] = key.split('-');
            return {
                x: [positions[source].x, positions[target].x],
                y: [positions[source].y, positions[target].y],
                mode: 'lines',
                type: 'scatter',
                line: {
                    width: Math.abs(value) * 5,
                    color: value >= 0 ? 'rgba(66,133,244,0.6)' : 'rgba(234,67,53,0.6)'
                },
                hoverinfo: 'text',
                hovertext: `${source} ↔ ${target}<br>Curvature: ${value.toFixed(3)}`
            };
        });

    return (
        <div>
            {renderFilters()}
            <Plot
                data={[...edgeTraces, nodeTrace]}
                layout={{
                    title: {
                        text: 'Stock Market Network<br>' +
                              `<span style="font-size:12px">Showing ${filteredNodes.length} nodes and ${Object.keys(filteredCurvature).length} edges</span>`,
                        font: { size: 18 }
                    },
                    showlegend: false,
                    hovermode: 'closest',
                    margin: { l: 20, r: 20, t: 60, b: 20 },
                    xaxis: { showgrid: false, zeroline: false, showticklabels: false },
                    yaxis: { showgrid: false, zeroline: false, showticklabels: false },
                    width: 800,
                    height: 600,
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)'
                }}
                config={{
                    displayModeBar: false,
                    responsive: true
                }}
            />

            {selectedNode && (
                <div style={styles.card}>
                    <h3>{selectedNode} Details</h3>
                    <p>Loading additional metrics...</p>
                    {/* Stock-specific details will be added here */}
                </div>
            )}
        </div>
    );
  };

  const renderPredictions = () => {
    if (!showPredictions || !predictions) return null;

    // Prepare data for the risk-return plot
    const riskReturnData = predictions.risk_return_analysis;
    
    // Create scatter plot trace for each pair
    const scatterTrace = {
      x: riskReturnData.map(d => d.risk),
      y: riskReturnData.map(d => d.return),
      mode: 'markers+text',
      type: 'scatter',
      text: riskReturnData.map(d => d.pair),
      textposition: 'top center',
      marker: {
        size: 10,
        color: riskReturnData.map(d => d.correlation),
        colorscale: 'Viridis',
        showscale: true,
        colorbar: {
          title: 'Correlation',
          thickness: 15,
          len: 0.5
        }
      },
      hovertemplate: 
        '<b>%{text}</b><br>' +
        'Risk: %{x:.2f}<br>' +
        'Return: %{y:.2f}<br>' +
        'Sharpe Ratio: %{customdata:.2f}<br>' +
        '<extra></extra>',
      customdata: riskReturnData.map(d => d.sharpe_ratio)
    };

    return (
      <div style={styles.analysisSection}>
        <h3 style={styles.sectionTitle}>Market Predictions</h3>
        
        {/* Risk-Return Analysis Plot */}
        <div style={styles.plotContainer}>
          <h4 style={styles.subsectionTitle}>Risk-Return Analysis</h4>
          <Plot
            data={[scatterTrace]}
            layout={{
              title: 'Risk-Return Analysis of Stock Pairs',
              xaxis: {
                title: 'Risk (Annualized Volatility)',
                gridcolor: 'rgba(0,0,0,0.1)'
              },
              yaxis: {
                title: 'Return (Annualized)',
                gridcolor: 'rgba(0,0,0,0.1)'
              },
              hovermode: 'closest',
              plot_bgcolor: 'rgba(0,0,0,0)',
              paper_bgcolor: 'rgba(0,0,0,0)',
              width: 800,
              height: 500,
              margin: { l: 50, r: 50, t: 50, b: 50 }
            }}
            config={{
              displayModeBar: false,
              responsive: true
            }}
          />
        </div>

        {/* Best Pairs Analysis */}
        <div style={styles.bestPairsContainer}>
          <h4 style={styles.subsectionTitle}>Top Performing Pairs</h4>
          <div style={styles.bestPairsGrid}>
            {riskReturnData.slice(0, 3).map((pair, index) => (
              <div key={index} style={styles.bestPairCard}>
                <h5>{pair.pair}</h5>
                <div style={styles.metricGrid}>
                  <div style={styles.metric}>
                    <span>Sharpe Ratio:</span>
                    <span>{pair.sharpe_ratio.toFixed(2)}</span>
                  </div>
                  <div style={styles.metric}>
                    <span>Return:</span>
                    <span>{(pair.return * 100).toFixed(2)}%</span>
                  </div>
                  <div style={styles.metric}>
                    <span>Risk:</span>
                    <span>{(pair.risk * 100).toFixed(2)}%</span>
                  </div>
                  <div style={styles.metric}>
                    <span>Correlation:</span>
                    <span>{(pair.correlation * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Original Predictions Grid */}
        <div style={styles.predictionsGrid}>
          {predictions.predictions.map((pred, index) => (
            <div key={index} style={styles.predictionCard}>
              <h4 style={styles.pairTitle}>{pred.pair}</h4>
              <div style={styles.predictionMetrics}>
                <div style={styles.metric}>
                  <span>Direction:</span>
                  <span style={{
                    color: pred.direction === 'up' ? '#34a853' : '#ea4335',
                    fontWeight: 'bold'
                  }}>
                    {pred.direction.toUpperCase()}
                  </span>
                </div>
                <div style={styles.metric}>
                  <span>Confidence:</span>
                  <span>{(pred.confidence * 100).toFixed(1)}%</span>
                </div>
                <div style={styles.metric}>
                  <span>Correlation:</span>
                  <span>{(pred.correlation * 100).toFixed(1)}%</span>
                </div>
                <div style={styles.metric}>
                  <span>Volatility:</span>
                  <span>{(pred.volatility * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderAnalysis = () => {
    if (!analysisResult) return null;

    return (
      <div id="analysisSection" style={styles.analysisContainer}>
        {/* Anomalies Section */}
        <div style={styles.analysisSection}>
          <h3 style={styles.sectionTitle}>Market Analysis</h3>
          {analysisResult.anomalies && analysisResult.anomalies.length > 0 ? (
            analysisResult.anomalies.map((anomaly, index) => (
              <div key={index} style={{
                ...styles.anomalyCard,
                borderColor: anomaly.severity === 'high' ? '#ea4335' : '#fbbc04'
              }}>
                <h4>{anomaly.edge}</h4>
                <p>Severity: {anomaly.severity}</p>
                <p>Z-Score: {anomaly.z_score.toFixed(2)}</p>
                <p>Current Value: {anomaly.current_value.toFixed(3)}</p>
                <p>Historical Mean: {anomaly.historical_mean.toFixed(3)}</p>
              </div>
            ))
          ) : (
            <p>No anomalies detected</p>
          )}
        </div>

        {renderPredictions()}

        {/* Optimal Pairs Section */}
        <div style={styles.analysisSection}>
          <h3 style={styles.sectionTitle}>Optimal Trading Pairs</h3>
          {analysisResult.optimal_pairs && analysisResult.optimal_pairs.length > 0 ? (
            analysisResult.optimal_pairs.map((pair, index) => (
              <div key={index} style={styles.pairCard}>
                <h4>{pair.pair}</h4>
                <div style={styles.pairMetrics}>
                  <div style={styles.metric}>
                    <span>Score:</span>
                    <span>{pair.score.toFixed(3)}</span>
                  </div>
                  <div style={styles.metric}>
                    <span>Curvature:</span>
                    <span>{pair.curvature.toFixed(3)}</span>
                  </div>
                  <div style={styles.metric}>
                    <span>Sharpe Ratio:</span>
                    <span>{pair.sharpe_ratio.toFixed(2)}</span>
                  </div>
                  <div style={styles.metric}>
                    <span>Expected Return:</span>
                    <span>{(pair.expected_return * 100).toFixed(2)}%</span>
                  </div>
                  <div style={styles.metric}>
                    <span>Volatility:</span>
                    <span>{(pair.avg_volatility * 100).toFixed(2)}%</span>
                  </div>
                  <div style={styles.metric}>
                    <span>Hedge Ratio:</span>
                    <span>{pair.hedge_ratio.toFixed(2)}</span>
                  </div>
                </div>
                <div style={styles.sectorInfo}>
                  Sectors: {pair.sectors.join(' ↔ ')}
                </div>
              </div>
            ))
          ) : (
            <div style={styles.noPairsCard}>
              <h4>No Optimal Trading Pairs Found</h4>
              <p>Possible reasons:</p>
              <ul>
                <li>High volatility between pairs (current market conditions may be unstable)</li>
                <li>Low correlation between stocks (assets may be too dissimilar)</li>
                <li>Insufficient historical data for reliable analysis</li>
                <li>Market cap or volume differences too large between pairs</li>
              </ul>
              <p>Suggestions:</p>
              <ul>
                <li>Try selecting stocks from similar sectors</li>
                <li>Include stocks with comparable market caps</li>
                <li>Extend the date range for more historical data</li>
                <li>Consider adding more liquid stocks to the analysis</li>
              </ul>
            </div>
          )}
        </div>

        {/* Lead-Lag Analysis Section */}
        {leadLagData && (
          <div id="leadLagSection" style={styles.analysisSection}>
            <h3 style={styles.sectionTitle}>Lead-Lag Analysis</h3>
            
            {/* Sector Leaders */}
            <div style={styles.subsection}>
              <h4 style={styles.subsectionTitle}>Sector Leaders</h4>
              <div style={styles.leadersGrid}>
                {Object.entries(leadLagData.sector_leaders).map(([sector, data]) => (
                  <div key={sector} style={styles.leaderCard}>
                    <h4>{sector}</h4>
                    <div style={styles.leaderMetrics}>
                      <div style={styles.metric}>
                        <span>Leader:</span>
                        <span>{data.stock}</span>
                      </div>
                      <div style={styles.metric}>
                        <span>Significance:</span>
                        <span>{(data.significance * 100).toFixed(1)}%</span>
                      </div>
                      <div style={styles.metric}>
                        <span>Average Lag:</span>
                        <span>{data.avg_lag} days</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Lead-Lag Relationships */}
            <div style={styles.subsection}>
              <h4 style={styles.subsectionTitle}>Significant Relationships</h4>
              <div style={styles.relationshipsGrid}>
                {leadLagData.lead_lag_relationships.map((rel, index) => (
                  <div key={index} style={styles.relationshipCard}>
                    <div style={styles.relationshipHeader}>
                      <span style={styles.leaderStock}>{rel.leader}</span>
                      <span style={styles.arrow}>➔</span>
                      <span style={styles.followerStock}>{rel.follower}</span>
                    </div>
                    <div style={styles.relationshipMetrics}>
                      <div style={styles.metric}>
                        <span>Lag:</span>
                        <span>{rel.lag_days} days</span>
                      </div>
                      <div style={styles.metric}>
                        <span>Correlation:</span>
                        <span>{(rel.correlation * 100).toFixed(1)}%</span>
                      </div>
                      <div style={styles.metric}>
                        <span>Significance:</span>
                        <span>{(rel.significance * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                    <div style={styles.sectorInfo}>
                      {rel.leader_sector} ➔ {rel.follower_sector}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderRegimeAnalysis = () => {
    if (!regimeData) return null;

    return (
      <div id="regimeSection" style={styles.analysisSection}>
        <h3 style={styles.sectionTitle}>Market Regime Analysis</h3>
        
        {/* Regime Timeline Plot */}
        <Plot
          data={[
            {
              x: regimeData.dates,
              y: regimeData.regimes,
              type: 'scatter',
              mode: 'lines',
              name: 'Market Regime',
              line: {
                color: regimeData.regimes.map(regime => 
                  regimeData.regime_types[regime] === 'bull' ? '#34a853' :
                  regimeData.regime_types[regime] === 'bear' ? '#ea4335' : '#fbbc04'
                )
              }
            }
          ]}
          layout={{
            title: 'Market Regime Timeline',
            xaxis: { title: 'Date' },
            yaxis: { 
              title: 'Regime',
              ticktext: Object.values(regimeData.regime_types),
              tickvals: Object.keys(regimeData.regime_types).map(Number)
            },
            height: 300,
            margin: { l: 50, r: 50, t: 50, b: 50 }
          }}
        />

        {/* Regime Statistics */}
        <div style={styles.regimeStats}>
          {Object.entries(regimeData.regime_stats).map(([regime, stats]) => (
            <div key={regime} style={{
              ...styles.regimeCard,
              backgroundColor: 
                regimeData.regime_types[regime] === 'bull' ? 'rgba(52, 168, 83, 0.1)' :
                regimeData.regime_types[regime] === 'bear' ? 'rgba(234, 67, 53, 0.1)' :
                'rgba(251, 188, 4, 0.1)'
            }}>
              <h4>{regimeData.regime_types[regime].toUpperCase()} Regime</h4>
              <div style={styles.regimeMetrics}>
                <div style={styles.metric}>
                  <span>Average Return:</span>
                  <span>{(stats.avg_return * 100).toFixed(2)}%</span>
                </div>
                <div style={styles.metric}>
                  <span>Volatility:</span>
                  <span>{(stats.volatility * 100).toFixed(2)}%</span>
                </div>
                <div style={styles.metric}>
                  <span>Duration:</span>
                  <span>{stats.duration} days</span>
                </div>
                <div style={styles.metric}>
                  <span>Stability:</span>
                  <span>{(regimeData.stability[regime] * 100).toFixed(2)}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Transition Matrix Heatmap */}
        <Plot
          data={[{
            z: regimeData.transition_matrix,
            x: Object.values(regimeData.regime_types),
            y: Object.values(regimeData.regime_types),
            type: 'heatmap',
            colorscale: 'Viridis'
          }]}
          layout={{
            title: 'Regime Transition Probabilities',
            height: 400,
            margin: { l: 100, r: 50, t: 50, b: 100 }
          }}
        />
      </div>
    );
  };

  // Helper function for force-directed layout
  const calculateForceDirectedLayout = (nodes, curvature) => {
    const positions = {};
    const iterations = 100;
    const k = 1; // Force strength

    // Initialize positions in a circle
    nodes.forEach((node, i) => {
      const angle = (2 * Math.PI * i) / nodes.length;
      positions[node] = {
        x: Math.cos(angle),
        y: Math.sin(angle),
        vx: 0,
        vy: 0
      };
    });

    // Run force simulation
    for (let i = 0; i < iterations; i++) {
      // Calculate forces
      nodes.forEach(node1 => {
        nodes.forEach(node2 => {
          if (node1 === node2) return;

          const dx = positions[node2].x - positions[node1].x;
          const dy = positions[node2].y - positions[node1].y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          // Repulsive force
          const repulsion = k / (distance * distance);
          positions[node1].vx -= dx * repulsion;
          positions[node1].vy -= dy * repulsion;

          // Attractive force based on curvature
          const edge = `${node1}-${node2}`;
          const reverseEdge = `${node2}-${node1}`;
          const curvatureValue = curvature[edge] || curvature[reverseEdge] || 0;
          
          if (curvatureValue !== 0) {
            const attraction = distance * Math.abs(curvatureValue) * k;
            positions[node1].vx += dx * attraction;
            positions[node1].vy += dy * attraction;
          }
        });
      });

      // Update positions
      nodes.forEach(node => {
        positions[node].x += positions[node].vx * 0.1;
        positions[node].y += positions[node].vy * 0.1;
        positions[node].vx *= 0.9;
        positions[node].vy *= 0.9;
      });
    }

    return positions;
  };

  // Update button styling to show disabled state more clearly
  const buttonStyle = (disabled) => ({
    ...styles.button,
    opacity: disabled ? 0.5 : 1,
    cursor: disabled ? 'not-allowed' : 'pointer'
  });

  // Update the date input to be modifiable
  const handleEndDateChange = (e) => {
    const selectedDate = e.target.value;
    if (selectedDate > today) {
      setEndDate(today);
    } else {
      setEndDate(selectedDate);
    }
  };

  // Ensure start date isn't after end date
  const handleStartDateChange = (e) => {
    const selectedDate = e.target.value;
    if (selectedDate > endDate) {
      setStartDate(endDate);
    } else {
      setStartDate(selectedDate);
    }
  };

  return (
    <>
      <div style={styles.background} />
      <div style={styles.container}>
        <h1 style={styles.title}>Stock Market Ricci Curvature Analysis</h1>
        
        <div style={styles.card}>
          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', color: '#2E3B4E' }}>
                Stock Tickers (comma-separated):
                <input
                  type="text"
                  value={tickers}
                  onChange={(e) => setTickers(e.target.value)}
                  placeholder="e.g., AAPL,MSFT,GOOGL,NVDA"
                  style={styles.input}
                />
              </label>
            </div>
            
            <div style={{ 
              display: 'flex', 
              gap: '20px', 
              marginBottom: '20px'
            }}>
              <label style={{ flex: 1 }}>
                <span style={{ color: '#2E3B4E' }}>Start Date:</span>
                <input
                  type="date"
                  value={startDate}
                  onChange={handleStartDateChange}
                  max={endDate}
                  style={styles.input}
                />
              </label>
              
              <label style={{ flex: 1 }}>
                <span style={{ color: '#2E3B4E' }}>End Date:</span>
                <input
                  type="date"
                  value={endDate}
                  onChange={handleEndDateChange}
                  max={today}
                  style={styles.input}
                />
              </label>
            </div>

            <div style={styles.buttonContainer}>
              <button 
                type="submit" 
                disabled={loading || !tickers.trim()}
                style={buttonStyle(loading || !tickers.trim())}
              >
                {loading ? 'Calculating...' : 'Calculate'}
              </button>
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={!result}
                style={{
                  ...buttonStyle(!result),
                  backgroundColor: '#34a853'
                }}
              >
                Analyze Market
              </button>
              <button
                type="button"
                onClick={handleRegimeAnalysis}
                disabled={!result}
                style={{
                  ...buttonStyle(!result),
                  backgroundColor: '#fbbc04'
                }}
              >
                Analyze Regimes
              </button>
              <button
                type="button"
                onClick={handleLeadLagAnalysis}
                disabled={!result}
                style={{
                  ...buttonStyle(!result),
                  backgroundColor: '#ea4335'
                }}
              >
                Analyze Lead-Lag
              </button>
              <button
                type="button"
                onClick={() => {
                  if (!showPredictions) {
                    handlePredictions();
                  } else {
                    setShowPredictions(false);
                  }
                }}
                disabled={!result}
                style={{
                  ...buttonStyle(!result),
                  backgroundColor: '#4285f4'
                }}
              >
                {showPredictions ? 'Hide Predictions' : 'Show Predictions'}
              </button>
            </div>
          </form>
        </div>

        {error && (
          <div style={{
            ...styles.card,
            color: '#ea4335',
            backgroundColor: 'rgba(253, 231, 231, 0.9)'
          }}>
            {error}
          </div>
        )}

        {result && (
          <div style={{
            ...styles.card,
            background: 'rgba(255, 255, 255, 0.95)'
          }}>
            <div style={{ marginBottom: '20px' }}>
              {renderGraph()}
            </div>
            <div style={{
              marginTop: '30px',
              padding: '20px',
              backgroundColor: 'rgba(248, 249, 250, 0.8)',
              borderRadius: '8px'
            }}>
              <h2 style={{ color: '#2E3B4E', marginBottom: '15px' }}>Raw Results:</h2>
              <pre style={{ 
                backgroundColor: 'rgba(248, 249, 250, 0.9)',
                padding: '15px',
                borderRadius: '8px',
                overflow: 'auto',
                fontSize: '14px',
                color: '#2E3B4E'
              }}>
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          </div>
        )}

        {analysisResult && renderAnalysis()}
        {showRegimes && renderRegimeAnalysis()}
      </div>
    </>
  );
}

export default App;
