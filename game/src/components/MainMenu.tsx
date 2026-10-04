import Link from "next/link";

import styles from "./main-menu.module.css";

export function MainMenu() {
  return (
    <div className={styles.hero}>
      <p className={styles.badge}>Motor imagery · BCI feedback</p>
      <h1 className={styles.title}>The Collector</h1>
      <p className={styles.lead}>
        Experimental 2D visual-feedback platform for offline cueing and online
        classifier-driven control. Phaser runs the trial loop; Next.js persists
        telemetry and session data.
      </p>
      <nav className={styles.nav}>
        <Link className={styles.primary} href="/game?step=offline">
          Offline
        </Link>
        <Link className={styles.primary} href="/game?step=simulator">
          Online simulated
        </Link>
        <Link className={styles.primary} href="/game?step=real">
          Online real
        </Link>
        <Link className={styles.secondary} href="/results">
          Sessions & export
        </Link>
      </nav>
    </div>
  );
}
