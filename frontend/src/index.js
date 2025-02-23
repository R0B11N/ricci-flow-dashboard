import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './index.css';
import App from './App';
import FutureRegimeAnalyzer from './components/FutureRegimeAnalyzer';

// Remove any existing content
const rootElement = document.getElementById('root');
while (rootElement.firstChild) {
  rootElement.removeChild(rootElement.firstChild);
}

// Create root and render
const root = createRoot(rootElement);
root.render(
  <Router>
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/future-regime" element={<FutureRegimeAnalyzer />} />
    </Routes>
  </Router>
);
