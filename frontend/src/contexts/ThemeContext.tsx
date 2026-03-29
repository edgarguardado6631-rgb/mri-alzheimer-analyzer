import React, { createContext, useContext, useState, useEffect } from 'react';

type ThemeContextType = {
    isDark: boolean;
    setIsDark: (dark: boolean) => void;
};

const ThemeContext = createContext<ThemeContextType>({
    isDark: true,
    setIsDark: () => { },
});

export const useTheme = () => useContext(ThemeContext);

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [isDark, setIsDark] = useState(true);

    useEffect(() => {
        // Check localStorage for saved preference
        const saved = localStorage.getItem('app-theme-dark');
        if (saved !== null) {
            setIsDark(saved === 'true');
        }
    }, []);

    const handleSetDark = (dark: boolean) => {
        setIsDark(dark);
        localStorage.setItem('app-theme-dark', String(dark));
    };

    return (
        <ThemeContext.Provider value={{ isDark, setIsDark: handleSetDark }}>
            {children}
        </ThemeContext.Provider>
    );
};
