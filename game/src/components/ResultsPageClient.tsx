"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { apiListSessions } from "@/lib/api/client";

import { ResultsTable, type SessionRow } from "./ResultsTable";
import styles from "./results-page.module.css";

export function ResultsPageClient() {
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void apiListSessions(50, 0)
      .then((r) => setSessions(r.sessions))
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : "Load failed"),
      );
  }, []);

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <Link href="/" className={styles.back}>
          ← Home
        </Link>
        <h1 className={styles.title}>Sessions</h1>
        <p className={styles.lead}>
          Persisted SQLite sessions. Export JSON or CSV for analysis.
        </p>
      </header>
      {error ? <p className={styles.error}>{error}</p> : null}
      <ResultsTable sessions={sessions} />
    </div>
  );
}
