import { createContext, useContext } from 'react';

export type ThemeContextType = {
  isDark: boolean;
  setIsDark: (dark: boolean) => void;
};

export const ThemeContext = createContext<ThemeContextType>({
  isDark: true,
  setIsDark: () => { },
});

export const useTheme = () => useContext(ThemeContext);
