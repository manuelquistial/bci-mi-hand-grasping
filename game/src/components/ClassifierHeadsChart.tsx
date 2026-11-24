"use client";

import { useEffect, useRef } from "react";

import type { PipelineStageMessage } from "@/lib/pipeline/observe";

import styles from "./monitor-page.module.css";

const HEAD_COLORS = ["#38bdf8", "#fbbf24", "#a3e635", "#f472b6"];

function shortName(classifier: string): string {
  if (classifier.includes("svm")) return "SVM";
  if (classifier.includes("logistic")) return "LR";
  return classifier;
}

function headValue(
  stage: PipelineStageMessage,
  classifier: string,
): number | null {
  const pred = stage.predictions.find((item) => item.classifier === classifier);
  if (!pred) return null;
  if (pred.p_left != null && pred.p_right != null) {
    return pred.p_right - pred.p_left; // -1 left … +1 right
  }
  if (pred.score == null) return null;
  if (pred.label === "left") return -Math.abs(pred.score);
  if (pred.label === "right") return Math.abs(pred.score);
  return pred.score;
}

export function ClassifierHeadsChart({
  stages,
}: {
  stages: PipelineStageMessage[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const classifiers = Array.from(
    new Set(stages.flatMap((stage) => stage.predictions.map((p) => p.classifier))),
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const height = 180;
    canvas.style.width = "100%";
    canvas.style.height = `${height}px`;
    const width = Math.max(220, canvas.clientWidth);
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#1c1917";
    context.fillRect(0, 0, width, height);

    const padL = 36;
    const padR = 12;
    const padT = 16;
    const padB = 28;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const midY = padT + plotH / 2;

    context.strokeStyle = "#44403c";
    context.beginPath();
    context.moveTo(padL, midY);
    context.lineTo(width - padR, midY);
    context.stroke();
    context.fillStyle = "#a8a29e";
    context.font = "10px system-ui, sans-serif";
    context.fillText("left −", padL, padT + 8);
    context.fillText("+ right", width - padR - 40, padT + 8);

    const xs = [0, 1, 2, 3, 4];
    xs.forEach((index) => {
      const x = padL + (index / 4) * plotW;
      context.strokeStyle = "#292524";
      context.beginPath();
      context.moveTo(x, padT);
      context.lineTo(x, padT + plotH);
      context.stroke();
      context.fillStyle = "#a8a29e";
      context.fillText(`w${index}`, x - 6, height - 10);
    });

    if (!stages.length || !classifiers.length) {
      context.fillStyle = "#78716c";
      context.fillText("Waiting for classified windows…", padL, midY - 8);
      return;
    }

    classifiers.forEach((classifier, order) => {
      const color = HEAD_COLORS[order % HEAD_COLORS.length]!;
      const points = stages
        .map((stage) => {
          const value = headValue(stage, classifier);
          if (value == null) return null;
          return { index: stage.window_index, value, label: stage.predictions.find((p) => p.classifier === classifier)?.label };
        })
        .filter((item): item is { index: number; value: number; label: string | null | undefined } => item != null);
      if (points.length === 0) return;
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.beginPath();
      points.forEach((point, i) => {
        const x = padL + (point.index / 4) * plotW;
        const y = midY - Math.max(-1, Math.min(1, point.value)) * (plotH * 0.42);
        if (i === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.stroke();
      points.forEach((point) => {
        const x = padL + (point.index / 4) * plotW;
        const y = midY - Math.max(-1, Math.min(1, point.value)) * (plotH * 0.42);
        context.fillStyle = color;
        context.beginPath();
        context.arc(x, y, 3.5, 0, Math.PI * 2);
        context.fill();
      });
      context.fillStyle = color;
      context.fillText(shortName(classifier), padL + order * 48, padT + 10);
    });
  }, [stages, classifiers]);

  const latest = stages[stages.length - 1];

  return (
    <figure className={`${styles.panel} ${styles.span2}`}>
      <figcaption>
        <span className={styles.title}>5 · Classifier heads</span>
        <span className={styles.detail}>
          {classifiers.map(shortName).join(" · ") || "SVM / LR"} · per window
          (not only the gated decision)
        </span>
      </figcaption>
      <canvas ref={canvasRef} aria-label="Classifier head scores" />
      <div className={styles.controlRow}>
        {(latest?.predictions ?? []).map((pred) => (
          <span key={pred.classifier}>
            {shortName(pred.classifier)}{" "}
            <strong>
              {pred.label ?? "—"}
              {pred.p_left != null && pred.p_right != null
                ? ` · P(L)=${pred.p_left.toFixed(2)} P(R)=${pred.p_right.toFixed(2)}`
                : pred.score != null
                  ? ` · score=${pred.score.toFixed(2)}`
                  : ""}
            </strong>
          </span>
        ))}
      </div>
    </figure>
  );
}
