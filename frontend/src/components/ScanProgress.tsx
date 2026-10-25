import React from 'react';
import { ProgressBar } from '@carbon/react';
import { CheckmarkFilled, CircleDash, InProgress } from '@carbon/icons-react';
import type { UploadProgress } from '../api';
import './ScanProgress.scss';

export interface ScanStep {
  label: string;
  /** Shown under the bar while this step is active. */
  detail?: string;
}

interface Props {
  steps: ScanStep[];
  /** Index of the step in progress. */
  active: number;
  /** Byte progress of the upload, shown while step 0 is active. */
  upload?: UploadProgress;
  fileName?: string;
}

const mb = (bytes: number) => (bytes / 1024 / 1024).toFixed(1);

/**
 * Progress panel for a scan upload: a real progress bar while the file is sent,
 * then an indeterminate bar while the server works, a step list and a live
 * elapsed-time counter, so long waits never look frozen.
 */
const ScanProgress: React.FC<Props> = ({ steps, active, upload, fileName }) => {
  const [elapsed, setElapsed] = React.useState(0);
  React.useEffect(() => {
    const start = Date.now();
    const id = window.setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => window.clearInterval(id);
  }, []);

  const uploading = active === 0 && upload && upload.total > 0;
  const percent = uploading ? Math.round((upload.loaded / upload.total) * 100) : undefined;
  const helper = uploading
    ? `${mb(upload.loaded)} of ${mb(upload.total)} MB · ${percent}%`
    : steps[active]?.detail ?? '';

  return (
    <div className="scan-progress" role="status" aria-live="polite">
      {fileName && <p className="scan-progress__file">{fileName}</p>}
      <ProgressBar
        label={steps[active]?.label ?? ''}
        helperText={helper}
        value={percent}
        max={100}
        size="big"
      />
      <ol className="scan-progress__steps">
        {steps.map((step, i) => (
          <li key={step.label} className={i < active ? 'is-done' : i === active ? 'is-active' : ''}>
            {i < active ? <CheckmarkFilled size={16} /> : i === active ? <InProgress size={16} /> : <CircleDash size={16} />}
            <span>{step.label}</span>
          </li>
        ))}
      </ol>
      <p className="scan-progress__elapsed">{elapsed} s elapsed</p>
    </div>
  );
};

export default ScanProgress;
