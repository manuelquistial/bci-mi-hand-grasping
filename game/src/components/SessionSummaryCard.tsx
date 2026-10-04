"use client";

import styles from "./session-summary-card.module.css";

type Props = {
  title?: string;
  trialCount: number;
  successCount: number;
  totalScore: number;
};

export function SessionSummaryCard({
  title = "Summary",
  trialCount,
  successCount,
  totalScore,
}: Props) {
  const rate = trialCount > 0 ? successCount / trialCount : 0;
  return (
    <section className={styles.card} aria-labelledby="summary-heading">
      <h2 id="summary-heading" className={styles.title}>
        {title}
      </h2>
      <dl className={styles.dl}>
        <div>
          <dt>Trials</dt>
          <dd>{trialCount}</dd>
        </div>
        <div>
          <dt>Successes</dt>
          <dd>{successCount}</dd>
        </div>
        <div>
          <dt>Success rate</dt>
          <dd>{(rate * 100).toFixed(1)}%</dd>
        </div>
        <div>
          <dt>Total score</dt>
          <dd>{totalScore}</dd>
        </div>
      </dl>
    </section>
  );
}
