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
