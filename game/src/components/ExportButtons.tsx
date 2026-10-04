"use client";

import styles from "./export-buttons.module.css";

type Props = {
  sessionId: string;
};

export function ExportButtons({ sessionId }: Props) {
  return (
    <div className={styles.actions}>
      <a
        className={styles.link}
        href={`/api/export/${sessionId}?format=json`}
        target="_blank"
        rel="noreferrer"
      >
        JSON
      </a>
      <a
        className={styles.link}
        href={`/api/export/${sessionId}?format=csv`}
        target="_blank"
        rel="noreferrer"
      >
        CSV
      </a>
    </div>
  );
}
