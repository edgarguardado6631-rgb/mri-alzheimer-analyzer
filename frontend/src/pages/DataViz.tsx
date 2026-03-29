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
  Tag,
  StructuredListWrapper,
  StructuredListBody,
  StructuredListRow,
  StructuredListCell,
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
    total_patients: 179,
    scans_processed: 0,
    model_accuracy: 0,
  });

  React.useEffect(() => {
    fetch(`${API_URL}/data/patients`)
      .then(res => res.json())
      .then(data => setPatients(data.patients))
      .catch(err => console.error('Failed to fetch patients:', err));

    fetch(`${API_URL}/data/stats`)
      .then(res => res.json())
      .then(data => setDashboardStats(data))
      .catch(err => console.error('Failed to fetch stats:', err));
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
                    <p className="cds--type-helper-text-01" style={{ marginTop: '0.75rem', color: 'var(--cds-text-secondary)' }}>Model Accuracy</p>
                    <p className="cds--type-productive-heading-06" style={{ marginTop: '0.25rem' }}>
                      {(dashboardStats.model_accuracy * 100).toFixed(1)}%
                    </p>
                    <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-secondary)', marginTop: '0.25rem' }}>Based on validation set</p>
                  </Tile>
                </Column>
                <Column lg={16} md={8} sm={4} style={{ marginTop: '1rem' }}>
                  <Tile style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <p className="cds--type-body-short-01" style={{ color: 'var(--cds-text-placeholder)' }}>
                      Cohort Demographics Chart — coming soon
                    </p>
                  </Tile>
                </Column>
              </Grid>
            </TabPanel>

            {/* MRI Viewer Tab */}
            <TabPanel>
              <Grid narrow style={{ marginTop: '1.5rem' }}>
                <Column lg={4} md={3} sm={4}>
                  <Layer>
                    <Tile style={{ padding: 0, overflow: 'hidden' }}>
                      <div style={{ padding: '1rem', borderBottom: '1px solid var(--cds-border-subtle-01)' }}>
                        <p className="cds--type-label-01" style={{ color: 'var(--cds-text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                          Patient List
                        </p>
                      </div>
                      {patients.length === 0 ? (
                        <div style={{ padding: '1rem' }}>
                          <p className="cds--type-helper-text-01" style={{ color: 'var(--cds-text-placeholder)' }}>No patients loaded</p>
                        </div>
                      ) : (
                        <StructuredListWrapper selection style={{ margin: 0 }}>
                          <StructuredListBody>
                            {patients.map(patient => (
                              <StructuredListRow
                                key={patient}
                                onClick={() => handlePatientSelect(patient)}
                                style={{
                                  cursor: 'pointer',
                                  background: selectedPatient === patient ? 'var(--cds-layer-selected-01)' : 'transparent',
                                }}
                              >
                                <StructuredListCell>
                                  <span className="cds--type-body-short-01">{patient}</span>
                                </StructuredListCell>
                              </StructuredListRow>
                            ))}
                          </StructuredListBody>
                        </StructuredListWrapper>
                      )}
                    </Tile>
                  </Layer>
                </Column>

                <Column lg={12} md={5} sm={4}>
                  <Tile
                    style={{
                      minHeight: '600px',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: 'var(--cds-background-inverse)',
                      padding: '1.5rem',
                    }}
                  >
                    {selectedPatient ? (
                      <>
                        {patientScans.length > 0 && (
                          <div style={{ width: '100%', marginBottom: '1rem' }}>
                            <Dropdown
                              id="scan-dropdown"
                              titleText="MRI Scan"
                              label="Choose a scan"
                              items={patientScans}
                              selectedItem={selectedScan}
                              onChange={handleScanSelect}
                            />
                          </div>
                        )}
                        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', minHeight: 0 }}>
                          {loading ? (
                            <InlineLoading description="Loading scan..." status="active" style={{ color: 'var(--cds-text-inverse)' }} />
                          ) : selectedScan ? (
                            <img
                              src={`${API_URL}/data/image/${selectedPatient}/${selectedScan}/slice/${currentSlice}`}
                              alt="MRI Slice"
                              style={{ maxHeight: '420px', maxWidth: '100%', objectFit: 'contain' }}
                            />
                          ) : (
                            <p className="cds--type-body-short-01" style={{ color: 'var(--cds-text-inverse)' }}>No scans available</p>
                          )}
                        </div>
                        <div style={{ width: '100%', paddingTop: '1rem' }}>
                          <Slider
                            labelText={`Slice: ${currentSlice} / ${maxSlices}`}
                            value={currentSlice}
                            min={0}
                            max={maxSlices}
                            step={1}
                            onChange={({ value }: { value: number }) => setCurrentSlice(value)}
                            hideTextInput
                          />
                        </div>
                      </>
                    ) : (
                      <div style={{ textAlign: 'center' }}>
                        <p className="cds--type-productive-heading-02" style={{ color: 'var(--cds-text-inverse)', marginBottom: '0.5rem' }}>
                          No patient selected
                        </p>
                        <p className="cds--type-body-short-01" style={{ color: 'var(--cds-text-secondary)' }}>
                          Choose a patient from the list to view their MRI scans
                        </p>
                      </div>
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
