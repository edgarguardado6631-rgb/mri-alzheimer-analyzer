import { Tile, Toggle, Select, SelectItem, Button } from '@carbon/react';
import { useTheme } from '../contexts/ThemeContext';

const Settings = () => {
    const { isDark, setIsDark } = useTheme();

    return (
        <div className="cds--grid" style={{ padding: '2rem' }}>
            <div className="cds--row">
                <div className="cds--col-lg-8 cds--offset-lg-4">
                    <h2 style={{ marginBottom: '2rem' }}>Settings</h2>
                    <Tile>
                        <h4 style={{ marginBottom: '1rem' }}>Application Preferences</h4>

                        <div style={{ marginBottom: '1.5rem' }}>
                            <Toggle
                                id="theme-toggle"
                                labelText="Dark Mode"
                                labelA="Off"
                                labelB="On"
                                toggled={isDark}
                                onToggle={(toggled) => setIsDark(toggled)}
                            />
                        </div>

                        <div style={{ marginBottom: '1.5rem' }}>
                            <Select id="model-select" labelText="AI Model Selection" helperText="Choose which model to use for analysis">
                                <SelectItem value="cnn-v1" text="Internal CNN v1.0 (Basic)" />
                                <SelectItem value="cnn-v2" text="Internal CNN v2.0 (Advanced)" />
                                <SelectItem value="external-gpt" text="External AI (GPT-4 Vision)" />
                            </Select>
                        </div>

                        <div style={{ marginBottom: '1.5rem' }}>
                            <p style={{ marginBottom: '0.5rem', fontWeight: 600 }}>Data Source</p>
                            <div style={{ padding: '1rem', background: 'var(--cds-layer-01)', fontFamily: 'monospace' }}>
                                e:/code/project/data/Test 2/ADNI
                            </div>
                        </div>

                        <Button>Save Changes</Button>
                    </Tile>
                </div>
            </div>
        </div>
    );
};

export default Settings;
