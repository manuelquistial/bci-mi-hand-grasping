import Link from "next/link";
import { redirect } from "next/navigation";

import { GamePageClient } from "@/components/GamePageClient";

import styles from "./game.module.css";

const steps = ["offline", "simulator", "signal", "real"] as const;

type Step = (typeof steps)[number];

function isStep(value: string | undefined): value is Step {
  return steps.some((step) => step === value);
}

export default async function GamePage({
  searchParams,
}: {
  searchParams: Promise<{ step?: string }>;
}) {
  const { step } = await searchParams;
  if (!isStep(step)) redirect("/");

  return (
    <div className={styles.wrap}>
      <header className={styles.header}>
        <Link href="/" className={styles.back}>
          Home
        </Link>
        <Link href="/results" className={styles.sub}>
          Results
        </Link>
      </header>
      <GamePageClient initialStep={step} />
    </div>
  );
}
