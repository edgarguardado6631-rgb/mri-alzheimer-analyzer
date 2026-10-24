import React from 'react';
import {
  Grid,
  Column,
  Tile,
  Toggle,
  Stack,
  FormGroup,
  Layer,
  Tag,
  InlineNotification,
} from '@carbon/react';
import { useTheme } from '../contexts/theme';
import { getJson } from '../api';
import type { ModelMetrics } from '../api';
import API_URL from '../config';

interface Health {
  status: string;
  model_loaded: boolean;
  model: string | null;
  task: string | null;
}

const Row = ({ label, value }: { label: string; value: React.ReactNode }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', padding: '0.5rem 0', borderBottom: '1px solid var(--cds-border-subtle-01)' }}>
    <span className="cds--type-body-short-01" style={{ color: 'var(--cds-text-secondary)' }}>{label}</span>
    <span className="cds--type-body-short-01" style={{ textAlign: 'right' }}>{value}</span>
  </div>
);

const Settings = () => {
  const { isDark, setIsDark } = useTheme();
  const [health, setHealth] = React.useState<Health | null>(null);
  const [metrics, setMetrics] = React.useState<ModelMetrics | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    getJson<Health>('/health').then(setHealth).catch(e => setError(e.message));
    getJson<ModelMetrics>('/data/model-metrics').then(setMetrics).catch(() => undefined);
  }, []);

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
          <Stack gap={6}>
            <Tile>
              <h4 className="cds--type-productive-heading-02" style={{ marginBottom: '1.5rem' }}>Appearance</h4>
              <FormGroup legendText="">
                <Toggle
                  id="theme-toggle"
                  labelText="Dark mode"
                  labelA="Light"
                  labelB="Dark"
                  toggled={isDark}
                  onToggle={(toggled) => setIsDark(toggled)}
                />
              </FormGroup>
            </Tile>

            <Tile>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <h4 className="cds--type-productive-heading-02">Deployed model</h4>
                {health && (
                  <Tag type={health.model_loaded ? 'green' : 'red'} size="sm">
                    {health.model_loaded ? 'Loaded' : 'Not loaded'}
                  </Tag>
                )}
              </div>

              {error && (
                <InlineNotification kind="error" title="Backend unreachable" subtitle={`${API_URL}: ${error}`} hideCloseButton lowContrast />
              )}

              {metrics && (
                <>
                  <Row label="Architecture" value={metrics.model} />
                  {metrics.task && <Row label="Task" value={metrics.task} />}
                  {metrics.cohort && <Row label="Training cohort" value={`${metrics.cohort.subjects} subjects (${metrics.cohort.CN} CN, ${metrics.cohort.AD} AD)`} />}
                  {metrics.cv_auc_mean !== undefined && (
                    <Row label="ROC AUC" value={`${metrics.cv_auc_mean.toFixed(3)} ± ${(metrics.cv_auc_std ?? 0).toFixed(3)}`} />
                  )}
                  <Row label="Accuracy" value={`${(metrics.accuracy * 100).toFixed(1)}%`} />
                  <Row label="Validation" value={`Patient-level ${metrics.cv_folds ?? 5}-fold cross-validation`} />
                  <Row label="Backend" value={<code>{API_URL}</code>} />
                </>
              )}

              <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)', marginTop: '1rem' }}>
                This is a research prototype. Its predictions support, but never replace, a clinical diagnosis.
              </p>
            </Tile>
          </Stack>
        </Layer>
      </Column>
    </Grid>
  );
};

export default Settings;
