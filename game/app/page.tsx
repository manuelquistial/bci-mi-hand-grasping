import { MainMenu } from "@/components/MainMenu";

import styles from "./page.module.css";

export default function Home() {
  return (
    <div className={styles.page}>
      <MainMenu />
    </div>
  );
}
