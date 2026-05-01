import React from 'react';
import {
  Grid,
  Column,
  Tile,
  Tabs,
  Tab,
  TabList,
  TabPanels,
  TabPanel,
  Slider,
  Dropdown,
  ComboBox,
  Tag,
  InlineLoading,
  Layer,
} from '@carbon/react';
import API_URL from '../config';

const DataViz = () => {
  const [patients, setPatients] = React.useState<string[]>([]);
  const [selectedPatient, setSelectedPatient] = React.useState<string | null>(null);
  const [patientScans, setPatientScans] = React.useState<string[]>([]);
  const [selectedScan, setSelectedScan] = React.useState<string | null>(null);
  const [currentSlice, setCurrentSlice] = React.useState<number>(0);
  const [maxSlices, setMaxSlices] = React.useState<number>(100);
  const [loading, setLoading] = React.useState<boolean>(false);
  const [dashboardStats, setDashboardStats] = React.useState({
    total_patients: 0,
    scans_processed: 0,
    model_accuracy: 0,
  });
  const [demographics, setDemographics] = React.useState<{
    groups: { label: string; count: number }[];
    sex: { label: string; count: number }[];
    age_bins: { label: string; count: number }[];
  }>({ groups: [], sex: [], age_bins: [] });
  const [modelMetrics, setModelMetrics] = React.useState<{
    accuracy: number;
    classes: { label: string; precision: number; recall: number; f1: number; support: number }[];
    macro_avg: { precision: number; recall: number; f1: number };
    weighted_avg: { precision: number; recall: number; f1: number };
    test_samples: number;
    epochs: number;
    model: string;
  } | null>(null);

  React.useEffect(() => {
    fetch(`${API_URL}/data/patients`)
      .then(res => res.json())
      .then(data => setPatients(data.patients))
      .catch(err => console.error('Failed to fetch patients:', err));

    fetch(`${API_URL}/data/stats`)
      .then(res => res.json())
      .then(data => setDashboardStats(data))
      .catch(err => console.error('Failed to fetch stats:', err));

    fetch(`${API_URL}/data/demographics`)
      .then(res => res.json())
      .then(data => setDemographics(data))
      .catch(err => console.error('Failed to fetch demographics:', err));

    fetch(`${API_URL}/data/model-metrics`)
      .then(res => res.json())
      .then(data => setModelMetrics(data))
      .catch(err => console.error('Failed to fetch model metrics:', err));
  }, []);

  const fetchMetadata = async (patientId: string, scanFilename: string) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/data/image/${patientId}/${scanFilename}/metadata`);
      const data = await res.json();
      setMaxSlices(data.max_slice);
      setCurrentSlice(Math.floor(data.max_slice / 2));
    } catch (err) {
      console.error('Failed to fetch metadata:', err);
    } finally {
      setLoading(false);
    }
  };

  const handlePatientSelect = async (patientId: string) => {
    setSelectedPatient(patientId);
    setLoading(true);
    try {
      const scansRes = await fetch(`${API_URL}/data/image/${patientId}/scans`);
      const scansData = await scansRes.json();
      setPatientScans(scansData.scans || []);
      if (scansData.scans && scansData.scans.length > 0) {
        const initialScan = scansData.scans[0];
        setSelectedScan(initialScan);
        await fetchMetadata(patientId, initialScan);
      } else {
        setSelectedScan(null);
        setLoading(false);
      }
    } catch (err) {
      console.error('Failed to fetch scans:', err);
      setLoading(false);
    }
  };

  const handleScanSelect = (item: { selectedItem: string }) => {
    const scan = item.selectedItem;
    setSelectedScan(scan);
    if (selectedPatient) fetchMetadata(selectedPatient, scan);
  };

  return (
    <Grid>
      <Column lg={16} md={8} sm={4}>
        <div style={{ marginBottom: '2rem' }}>
          <p className="cds--label" style={{ color: 'var(--cds-link-primary)', fontWeight: 500, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
            Clinical Intelligence
          </p>
          <h1 className="cds--type-productive-heading-05">Data Visualization & Analysis</h1>
        </div>
      </Column>

      <Column lg={16} md={8} sm={4}>
        <Tabs>
          <TabList aria-label="Analysis tabs">
            <Tab>Dashboard</Tab>
            <Tab>MRI Viewer</Tab>
          </TabList>
          <TabPanels>

            {/* Dashboard Tab */}
            <TabPanel>
              <Grid narrow style={{ marginTop: '1.5rem' }}>
                <Column lg={4} md={4} sm={4}>
                  <Tile>
                    <Tag type="blue" size="sm">Cohort</Tag>
                    <p className="cds--type-helper-text-01" style={{ marginTop: '0.75rem', color: 'var(--cds-text-secondary)' }}>Total Patients</p>
                    <p className="cds--type-productive-heading-06" style={{ marginTop: '0.25rem' }}>{dashboardStats.total_patients}</p>
                  </Tile>
                </Column>
                <Column lg={4} md={4} sm={4}>
                  <Tile>
                    <Tag type="teal" size="sm">Processed</Tag>
                    <p className="cds--type-helper-text-01" style={{ marginTop: '0.75rem', color: 'var(--cds-text-secondary)' }}>Scans Analyzed</p>
                    <p className="cds--type-productive-heading-06" style={{ marginTop: '0.25rem' }}>{dashboardStats.scans_processed.toLocaleString()}</p>
                  </Tile>
                </Column>
                <Column lg={8} md={8} sm={4}>
                  <Tile>
                    <Tag type="green" size="sm">CNN Model</Tag>
                    <p className="cds--type-helper-text-01" style={{ marginTop: '0.75rem', color: 'var(--cds-text-secondary)' }}>Overall Accuracy</p>
                    <p className="cds--type-productive-heading-06" style={{ marginTop: '0.25rem' }}>
                      {(dashboardStats.model_accuracy * 100).toFixed(1)}%
                    </p>
                    {modelMetrics && (
                      <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)', marginTop: '0.25rem' }}>
                        {modelMetrics.model} · {modelMetrics.epochs} epochs · {modelMetrics.test_samples} test samples
                      </p>
                    )}
                  </Tile>
                </Column>
                <Column lg={16} md={8} sm={4} style={{ marginTop: '1rem' }}>
                  <Tile>
                    <p className="cds--type-productive-heading-02" style={{ marginBottom: '1.5rem' }}>Cohort Demographics</p>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '2rem' }}>

                      {/* Diagnosis group bars */}
                      <div>
                        <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Diagnosis Group</p>
                        {(() => {
                          const max = Math.max(...demographics.groups.map(g => g.count), 1);
                          const colors: Record<string, string> = { CN: 'var(--cds-support-success)', MCI: 'var(--cds-support-warning)', AD: 'var(--cds-support-error)' };
                          return demographics.groups.map(g => (
                            <div key={g.label} style={{ marginBottom: '0.75rem' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                <span className="cds--type-body-short-01">{g.label}</span>
                                <span className="cds--type-body-short-01" style={{ color: 'var(--cds-text-secondary)' }}>{g.count}</span>
                              </div>
                              <div style={{ height: '8px', background: 'var(--cds-layer-02)', borderRadius: '2px' }}>
                                <div style={{ height: '100%', width: `${(g.count / max) * 100}%`, background: colors[g.label] ?? 'var(--cds-link-primary)', borderRadius: '2px', transition: 'width 0.6s ease' }} />
                              </div>
                            </div>
                          ));
                        })()}
                      </div>

                      {/* Sex bars */}
                      <div>
                        <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Sex</p>
                        {(() => {
                          const max = Math.max(...demographics.sex.map(s => s.count), 1);
                          return demographics.sex.map(s => (
                            <div key={s.label} style={{ marginBottom: '0.75rem' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                <span className="cds--type-body-short-01">{s.label === 'M' ? 'Male' : s.label === 'F' ? 'Female' : s.label}</span>
                                <span className="cds--type-body-short-01" style={{ color: 'var(--cds-text-secondary)' }}>{s.count}</span>
                              </div>
                              <div style={{ height: '8px', background: 'var(--cds-layer-02)', borderRadius: '2px' }}>
                                <div style={{ height: '100%', width: `${(s.count / max) * 100}%`, background: 'var(--cds-link-primary)', borderRadius: '2px', transition: 'width 0.6s ease' }} />
                              </div>
                            </div>
                          ));
                        })()}
                      </div>

                      {/* Age distribution bars */}
                      <div>
                        <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Age Distribution</p>
                        {(() => {
                          const max = Math.max(...demographics.age_bins.map(b => b.count), 1);
                          return demographics.age_bins.map(b => (
                            <div key={b.label} style={{ marginBottom: '0.75rem' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                                <span className="cds--type-body-short-01">{b.label}</span>
                                <span className="cds--type-body-short-01" style={{ color: 'var(--cds-text-secondary)' }}>{b.count}</span>
                              </div>
                              <div style={{ height: '8px', background: 'var(--cds-layer-02)', borderRadius: '2px' }}>
                                <div style={{ height: '100%', width: `${(b.count / max) * 100}%`, background: 'var(--cds-link-secondary)', borderRadius: '2px', transition: 'width 0.6s ease' }} />
                              </div>
                            </div>
                          ));
                        })()}
                      </div>

                    </div>
                  </Tile>
                </Column>

                {modelMetrics && (
                  <Column lg={16} md={8} sm={4} style={{ marginTop: '1rem' }}>
                    <Tile>
                      <p className="cds--type-productive-heading-02" style={{ marginBottom: '0.25rem' }}>Per-Class Performance</p>
                      <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)', marginBottom: '1.5rem' }}>
                        Classification report — {modelMetrics.model} · {modelMetrics.epochs} epochs · {modelMetrics.test_samples} test samples
                      </p>

                      {/* Per-class expanded rows */}
                      {(() => {
                        const classColors: Record<string, string> = {
                          CN: 'var(--cds-support-success)',
                          MCI: 'var(--cds-support-warning)',
                          AD: 'var(--cds-support-error)',
                        };
                        return modelMetrics.classes.map((cls) => {
                          const color = classColors[cls.label] ?? 'var(--cds-link-primary)';
                          return (
                            <div key={cls.label} style={{ marginBottom: '1.5rem', paddingBottom: '1.5rem', borderBottom: '1px solid var(--cds-border-subtle-01)' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                                <span className="cds--type-body-short-02" style={{ fontWeight: 600, color, minWidth: '3rem' }}>{cls.label}</span>
                                <Tag type="gray" size="sm">{cls.support} samples</Tag>
                              </div>
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.25rem' }}>
                                {[
                                  { label: 'Precision', value: cls.precision },
                                  { label: 'Recall', value: cls.recall },
                                  { label: 'F1 Score', value: cls.f1 },
                                ].map(({ label, value }) => (
                                  <div key={label}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.375rem' }}>
                                      <span className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label}</span>
                                      <span className="cds--type-body-short-01" style={{ fontWeight: 600 }}>{(value * 100).toFixed(1)}%</span>
                                    </div>
                                    <div style={{ height: '6px', background: 'var(--cds-layer-02)', borderRadius: '3px' }}>
                                      <div style={{ height: '100%', width: `${value * 100}%`, background: color, borderRadius: '3px', transition: 'width 0.6s ease' }} />
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        });
                      })()}

                      {/* Averages footer */}
                      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr', gap: '0.5rem', paddingTop: '0.5rem', paddingBottom: '0.5rem', borderBottom: '1px solid var(--cds-border-subtle-01)' }}>
                        {['', 'Precision', 'Recall', 'F1', 'Support'].map(h => (
                          <p key={h} className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{h}</p>
                        ))}
                      </div>
                      {[
                        { label: 'Macro avg', ...modelMetrics.macro_avg },
                        { label: 'Weighted avg', ...modelMetrics.weighted_avg },
                      ].map(row => (
                        <div key={row.label} style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr', gap: '0.5rem', padding: '0.5rem 0', borderBottom: '1px solid var(--cds-border-subtle-00)' }}>
                          <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)' }}>{row.label}</p>
                          <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)' }}>{row.precision.toFixed(2)}</p>
                          <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)' }}>{row.recall.toFixed(2)}</p>
                          <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)' }}>{row.f1.toFixed(2)}</p>
                          <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)' }}>{modelMetrics.test_samples}</p>
                        </div>
                      ))}
                    </Tile>
                  </Column>
                )}
              </Grid>
            </TabPanel>

            {/* MRI Viewer Tab */}
            <TabPanel>
              <Grid narrow style={{ marginTop: '1.5rem' }}>

                {/* Controls row */}
                <Column lg={6} md={4} sm={4}>
                  <Layer>
                    <ComboBox
                      id="patient-search"
                      titleText="Patient"
                      placeholder="Search by patient ID…"
                      helperText={patients.length > 0 ? `${patients.length} patients available` : 'Loading…'}
                      items={patients}
                      itemToString={(item) => item ?? ''}
                      selectedItem={selectedPatient}
                      onChange={({ selectedItem }) => {
                        if (selectedItem) handlePatientSelect(selectedItem);
                      }}
                    />
                  </Layer>
                </Column>

                <Column lg={6} md={4} sm={4}>
                  <Layer>
                    <Dropdown
                      id="scan-dropdown"
                      titleText="MRI Scan"
                      label={selectedPatient ? 'Choose a scan' : 'Select a patient first'}
                      items={patientScans}
                      selectedItem={selectedScan}
                      onChange={handleScanSelect}
                      disabled={!selectedPatient || patientScans.length === 0}
                    />
                  </Layer>
                </Column>

                {selectedPatient && selectedScan && (
                  <Column lg={4} md={8} sm={4} style={{ display: 'flex', alignItems: 'flex-end', paddingBottom: '0.25rem' }}>
                    <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                      <Tag type="blue" size="sm">{selectedPatient}</Tag>
                      <Tag type="gray" size="sm">Slice {currentSlice}/{maxSlices}</Tag>
                    </div>
                  </Column>
                )}

                {/* Viewer */}
                <Column lg={16} md={8} sm={4} style={{ marginTop: '1rem' }}>
                  <Tile
                    style={{
                      minHeight: '560px',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: 'var(--cds-background-inverse)',
                      padding: '1.5rem',
                    }}
                  >
                    {!selectedPatient ? (
                      <div style={{ textAlign: 'center' }}>
                        <p className="cds--type-productive-heading-02" style={{ color: 'var(--cds-text-inverse)', marginBottom: '0.5rem' }}>
                          No patient selected
                        </p>
                        <p className="cds--type-body-short-01" style={{ color: 'var(--cds-text-placeholder)' }}>
                          Search and select a patient above to view their MRI scans
                        </p>
                      </div>
                    ) : (
                      <>
                        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', minHeight: 0 }}>
                          {loading ? (
                            <InlineLoading description="Loading scan…" status="active" style={{ color: 'var(--cds-text-inverse)' }} />
                          ) : selectedScan ? (
                            <img
                              src={`${API_URL}/data/image/${selectedPatient}/${selectedScan}/slice/${currentSlice}`}
                              alt="MRI Slice"
                              style={{ maxHeight: '440px', maxWidth: '100%', objectFit: 'contain' }}
                            />
                          ) : (
                            <p className="cds--type-body-short-01" style={{ color: 'var(--cds-text-placeholder)' }}>
                              No scans available for this patient
                            </p>
                          )}
                        </div>
                        {selectedScan && (
                          <div style={{ width: '100%', maxWidth: '40rem', paddingTop: '1.5rem' }}>
                            <Slider
                              labelText={`Slice index`}
                              value={currentSlice}
                              min={0}
                              max={maxSlices}
                              step={1}
                              onChange={({ value }: { value: number }) => setCurrentSlice(value)}
                            />
                          </div>
                        )}
                      </>
                    )}
                  </Tile>
                </Column>

              </Grid>
            </TabPanel>

          </TabPanels>
        </Tabs>
      </Column>
    </Grid>
  );
};

export default DataViz;
