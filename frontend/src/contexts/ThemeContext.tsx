import React, { useState } from 'react';
import { ThemeContext } from './theme';

const STORAGE_KEY = 'app-theme-dark';

const readSavedTheme = (): boolean => {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved === null ? true : saved === 'true';
  } catch {
    return true;            // storage blocked (private window, etc.)
  }
};

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isDark, setIsDark] = useState(readSavedTheme);

  const handleSetDark = (dark: boolean) => {
    setIsDark(dark);
    try {
      localStorage.setItem(STORAGE_KEY, String(dark));
    } catch {
      // preference simply won't persist
    }
  };

  return (
    <ThemeContext.Provider value={{ isDark, setIsDark: handleSetDark }}>
      {children}
    </ThemeContext.Provider>
  );
};
