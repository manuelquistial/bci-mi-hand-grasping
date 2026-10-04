import Link from "next/link";

import { GamePageClient } from "@/components/GamePageClient";

import styles from "./game.module.css";

export default function GamePage() {
  return (
    <div className={styles.wrap}>
      <header className={styles.header}>
        <Link href="/" className={styles.back}>
          ← Home
        </Link>
        <Link href="/results" className={styles.sub}>
          Results
        </Link>
      </header>
      <GamePageClient />
    </div>
  );
}
