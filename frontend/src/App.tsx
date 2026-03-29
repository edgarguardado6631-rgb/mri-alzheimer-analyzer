import { Routes, Route } from 'react-router-dom';
import { Theme } from '@carbon/react';
import AppHeader from './components/AppHeader';
import Welcome from './pages/Welcome';
import DataViz from './pages/DataViz';
import Settings from './pages/Settings';
import { useTheme } from './contexts/ThemeContext';
import './global.scss';

const App = () => {
  const { isDark } = useTheme();

  return (
    <Theme theme={isDark ? 'g100' : 'white'} style={{ minHeight: '100vh', backgroundColor: 'var(--cds-background)' }}>
      <AppHeader />
      <div style={{ marginTop: '3rem' }}>
        <Routes>
          <Route path="/" element={<Welcome />} />
          <Route path="/visualization" element={<DataViz />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </div>
    </Theme>
  );
};

export default App;
