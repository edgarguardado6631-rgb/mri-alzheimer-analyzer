import {
  Grid,
  Column,
  Tile,
  Toggle,
  Select,
  SelectItem,
  Button,
  Stack,
  FormGroup,
  CodeSnippet,
  Layer,
} from '@carbon/react';
import { useTheme } from '../contexts/ThemeContext';

const Settings = () => {
  const { isDark, setIsDark } = useTheme();

  return (
    <Grid>
      <Column lg={16} md={8} sm={4}>
        <div style={{ marginBottom: '2rem' }}>
          <p className="cds--label" style={{ color: 'var(--cds-link-primary)', fontWeight: 500, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
            Configuration
          </p>
          <h1 className="cds--type-productive-heading-05">Settings</h1>
        </div>
      </Column>

      <Column lg={{ span: 8, offset: 4 }} md={8} sm={4}>
        <Layer>
          <Tile>
            <Stack gap={7}>
              <div>
                <h4 className="cds--type-productive-heading-02" style={{ marginBottom: '1.5rem' }}>
                  Application Preferences
                </h4>

                <Stack gap={6}>
                  <FormGroup legendText="Appearance">
                    <Toggle
                      id="theme-toggle"
                      labelText="Dark Mode"
                      labelA="Light"
                      labelB="Dark"
                      toggled={isDark}
                      onToggle={(toggled) => setIsDark(toggled)}
                    />
                  </FormGroup>

                  <Select
                    id="model-select"
                    labelText="AI Model"
                    helperText="Choose which model to use for MRI analysis"
                  >
                    <SelectItem value="cnn-v1" text="Internal CNN v1.0 — Basic" />
                    <SelectItem value="cnn-v2" text="Internal CNN v2.0 — Advanced" />
                    <SelectItem value="external-gpt" text="External AI — GPT-4 Vision" />
                  </Select>

                  <FormGroup legendText="Data Source">
                    <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)', marginBottom: '0.5rem' }}>
                      Active dataset path
                    </p>
                    <CodeSnippet type="single" feedback="Copied!">
                      data/Test 2/ADNI
                    </CodeSnippet>
                  </FormGroup>
                </Stack>
              </div>

              <div>
                <Button>Save Changes</Button>
              </div>
            </Stack>
          </Tile>
        </Layer>
      </Column>
    </Grid>
  );
};

export default Settings;
