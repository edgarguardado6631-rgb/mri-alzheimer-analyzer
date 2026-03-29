
import React from 'react';
import { Tile, Tabs, Tab, TabList, TabPanels, TabPanel, Slider, Dropdown } from '@carbon/react';
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
        model_accuracy: 0
    });

    React.useEffect(() => {
        // Fetch Patients List
        fetch(`${API_URL}/data/patients`)
            .then(res => res.json())
            .then(data => setPatients(data.patients))
            .catch(err => console.error("Failed to fetch patients:", err));

        // Fetch Dashboard Stats
        fetch(`${API_URL}/data/stats`)
            .then(res => res.json())
            .then(data => setDashboardStats(data))
            .catch(err => console.error("Failed to fetch stats:", err));
    }, []);

    const fetchMetadata = async (patientId: string, scanFilename: string) => {
        setLoading(true);
        try {
            const res = await fetch(`${API_URL}/data/image/${patientId}/${scanFilename}/metadata`);
            const data = await res.json();
            setMaxSlices(data.max_slice);
            setCurrentSlice(Math.floor(data.max_slice / 2));
        } catch (err) {
            console.error("Failed to fetch metadata:", err);
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
            console.error("Failed to fetch scans:", err);
            setLoading(false);
        }
    };

    const handleScanSelect = (item: { selectedItem: string }) => {
        const scan = item.selectedItem;
        setSelectedScan(scan);
        if (selectedPatient) {
            fetchMetadata(selectedPatient, scan);
        }
    };

    return (
        <div className="cds--grid" style={{ padding: '2rem' }}>
            <div className="cds--row">
                <div className="cds--col-lg-16">
                    <h2 style={{ marginBottom: '2rem' }}>Data Visualization & Analysis</h2>
                </div>
            </div>

            <Tabs>
                <TabList aria-label="List of tabs">
                    <Tab>Dashboard</Tab>
                    <Tab>MRI Viewer</Tab>
                </TabList>
                <TabPanels>
                    <TabPanel>
                        <div className="cds--row">
                            <div className="cds--col-lg-4">
                                <Tile>
                                    <h4>Total Patients</h4>
                                    <h1>{dashboardStats.total_patients}</h1>
                                </Tile>
                            </div>
                            <div className="cds--col-lg-4">
                                <Tile>
                                    <h4>Scans Processed</h4>
                                    <h1>{dashboardStats.scans_processed.toLocaleString()}</h1>
                                </Tile>
                            </div>
                            <div className="cds--col-lg-8">
                                <Tile>
                                    <h4>Model Accuracy</h4>
                                    <h1>{(dashboardStats.model_accuracy * 100).toFixed(1)}%</h1>
                                    <p>Based on validation set</p>
                                </Tile>
                            </div>
                        </div>
                        <div className="cds--row" style={{ marginTop: '1rem' }}>
                            <div className="cds--col-lg-16">
                                <Tile style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <p style={{ color: 'var(--cds-text-secondary)' }}>[Cohort Demographics Chart Placeholder]</p>
                                </Tile>
                            </div>
                        </div>
                    </TabPanel>
                    <TabPanel>
                        <div className="cds--row">
                            <div className="cds--col-lg-4">
                                <Tile>
                                    <h5>Select Patient</h5>
                                    <ul style={{ listStyle: 'none', padding: 0, marginTop: '1rem', maxHeight: '400px', overflowY: 'auto' }}>
                                        {patients.map(patient => (
                                            <li
                                                key={patient}
                                                onClick={() => handlePatientSelect(patient)}
                                                style={{
                                                    padding: '0.5rem',
                                                    cursor: 'pointer',
                                                    background: selectedPatient === patient ? 'var(--cds-layer-selected)' : 'transparent',
                                                    color: selectedPatient === patient ? 'var(--cds-text-primary)' : 'var(--cds-text-secondary)'
                                                }}
                                            >
                                                {patient}
                                            </li>
                                        ))}
                                    </ul>
                                </Tile>
                            </div>
                            <div className="cds--col-lg-12">
                                <Tile style={{ height: '600px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#000' }}>
                                    {selectedPatient ? (
                                        <>
                                            {patientScans.length > 0 && (
                                                <div style={{ width: '80%', marginBottom: '1rem', marginTop: '1rem' }}>
                                                    <Dropdown
                                                        id="scan-dropdown"
                                                        titleText="Select MRI Scan"
                                                        label="Choose a scan"
                                                        items={patientScans}
                                                        selectedItem={selectedScan}
                                                        onChange={handleScanSelect}
                                                    />
                                                </div>
                                            )}
                                            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', minHeight: 0 }}>
                                                {loading ? (
                                                    <p style={{ color: 'white' }}>Loading metadata...</p>
                                                ) : selectedScan ? (
                                                    <img
                                                        src={`${API_URL}/data/image/${selectedPatient}/${selectedScan}/slice/${currentSlice}`}
                                                        alt="MRI Slice"
                                                        style={{ maxHeight: '450px', maxWidth: '100%', objectFit: 'contain' }}
                                                    />
                                                ) : (
                                                    <p style={{ color: 'white' }}>No scans available for this patient</p>
                                                )}
                                            </div>
                                            <div style={{ width: '80%', padding: '1rem' }}>
                                                <Slider
                                                    labelText="Slice Index"
                                                    value={currentSlice}
                                                    min={0}
                                                    max={maxSlices}
                                                    step={1}
                                                    onChange={({ value }: { value: number }) => setCurrentSlice(value)}
                                                    hideTextInput
                                                />
                                            </div>
                                            <p style={{ color: 'white' }}>Slice: {currentSlice} / {maxSlices}</p>
                                        </>
                                    ) : (
                                        <p style={{ color: 'white' }}>Select a patient to view MRI scans</p>
                                    )}
                                </Tile>
                            </div>
                        </div>
                    </TabPanel>
                </TabPanels>
            </Tabs>
        </div>
    );
};

export default DataViz;
