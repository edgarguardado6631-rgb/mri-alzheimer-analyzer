import API_URL from './config';

/** Fetch JSON from the backend and throw the server's error detail on failure. */
export async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

interface PredictionJob<T> {
  job_id?: string;
  status: 'queued' | 'running' | 'done' | 'error';
  result?: T;
  detail?: string;
}

/**
 * Upload a scan and wait for its result. The server answers at once with a job
 * ID and analyzes the scan in the background (registration takes 15 to 60 s,
 * close to CloudFront's origin timeout), so the result is polled.
 */
export async function runPrediction<T>(form: FormData, pollMs = 2000, timeoutMs = 600000): Promise<T> {
  const { job_id } = await getJson<PredictionJob<T>>('/predict', { method: 'POST', body: form });
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, pollMs));
    const job = await getJson<PredictionJob<T>>(`/predict/${job_id}`);
    if (job.status === 'done' && job.result) return job.result;
    if (job.status === 'error') throw new Error(job.detail ?? 'Prediction failed.');
  }
  throw new Error('The analysis is taking too long. Please try again later.');
}

export interface DatasetStats {
  total_patients: number;
  total_scans: number;
  training_subjects: number;
  model_accuracy: number;
  model_auc: number;
}

export interface ClassMetrics {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelMetrics {
  model: string;
  task?: string;
  accuracy: number;
  cv_accuracy_mean?: number;
  cv_accuracy_std?: number;
  cv_auc_mean?: number;
  cv_auc_std?: number;
  roc_auc?: number;
  cv_folds?: number;
  epochs: number;
  test_samples: number;
  classes: ClassMetrics[];
  macro_avg?: { precision: number; recall: number; f1: number };
  weighted_avg?: { precision: number; recall: number; f1: number };
  confusion_matrix?: { labels: string[]; matrix: number[][] };
  preprocessing?: string[];
  cohort?: { subjects: number; CN: number; AD: number };
}

/** Short class code ("CN", "AD", "MCI") from a label like "CN (Cognitively Normal)". */
export const classCode = (label: string) => label.split(/[\s(,]/)[0];

export const CLASS_COLORS: Record<string, string> = {
  CN: 'var(--cds-support-success)',
  MCI: 'var(--cds-support-warning)',
  AD: 'var(--cds-support-error)',
};
