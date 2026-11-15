"use client";

import type { ReactNode } from "react";

import styles from "../settings-panel.module.css";

export function SettingsShell({
  title,
  help,
  children,
}: {
  title: string;
  help?: string;
  children: ReactNode;
}) {
  return (
    <section className={styles.panel} aria-labelledby="settings-heading">
      <h2 id="settings-heading" className={styles.title}>
        {title}
      </h2>
      {help ? <p className={styles.help}>{help}</p> : null}
      {children}
    </section>
  );
}

export function SettingsFieldset({
  legend,
  children,
}: {
  legend: string;
  children: ReactNode;
}) {
  return (
    <fieldset className={styles.fieldset}>
      <legend>{legend}</legend>
      {children}
    </fieldset>
  );
}
