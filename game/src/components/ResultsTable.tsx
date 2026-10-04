"use client";

import { ExportButtons } from "@/components/ExportButtons";

import styles from "./results-table.module.css";

export type SessionRow = {
  id: string;
  mode: string;
  startedAt: string;
  endedAt: string | null;
  totalScore: number;
  successRate: number | null;
  createdAt: string;
};

type Props = {
  sessions: SessionRow[];
};

export function ResultsTable({ sessions }: Props) {
  if (sessions.length === 0) {
    return (
      <p className={styles.empty}>No sessions yet. Run an experiment first.</p>
    );
  }

  return (
    <div className={styles.wrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Mode</th>
            <th>Started</th>
            <th>Ended</th>
            <th>Score</th>
            <th>Success %</th>
            <th>Export</th>
          </tr>
        </thead>
        <tbody>
          {sessions.map((s) => (
            <tr key={s.id}>
              <td>{s.mode}</td>
              <td>{new Date(s.startedAt).toLocaleString()}</td>
              <td>{s.endedAt ? new Date(s.endedAt).toLocaleString() : "—"}</td>
              <td>{s.totalScore}</td>
              <td>
                {s.successRate != null
                  ? `${(s.successRate * 100).toFixed(0)}%`
                  : "—"}
              </td>
              <td>
                <ExportButtons sessionId={s.id} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
