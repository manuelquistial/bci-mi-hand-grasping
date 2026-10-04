"use client";

import { useEffect, type MutableRefObject } from "react";

import {
  apiFinalizeSession,
  apiPostTelemetry,
  apiPostTrial,
} from "@/lib/api/client";
import type { GameBridge } from "@/game/bridge/game-bridge";
import { useHudStore } from "@/state/hud-store";
import { usePersistenceStore } from "@/state/persistence-store";

export function useGameBridgeEvents(
  bridge: GameBridge,
  sessionIdRef: MutableRefObject<string | null>,
) {
  const setLastError = usePersistenceStore((s) => s.setLastError);

  useEffect(() => {
    const unsub = bridge.subscribe((ev) => {
      if (ev.type === "SESSION_STARTED") {
        sessionIdRef.current = ev.sessionId;
        useHudStore.setState({
          sessionId: ev.sessionId,
          mode: ev.mode,
          sessionStatus: "running",
          bciStatusLine: null,
        });
        setLastError(null);
      }
      if (ev.type === "SESSION_PAUSED") {
        useHudStore.setState({ sessionStatus: "paused" });
      }
      if (ev.type === "SESSION_RESUMED") {
        useHudStore.setState({ sessionStatus: "running" });
      }
      if (ev.type === "SESSION_STOPPED") {
        useHudStore.setState({ sessionStatus: "idle", bciStatusLine: null });
        sessionIdRef.current = null;
      }
      if (ev.type === "PHASE_CHANGED") {
        useHudStore.setState({
          phase: ev.phase,
          trialIndex: ev.trialIndex,
          blockIndex: ev.blockIndex,
        });
      }
      if (ev.type === "TRIAL_STARTED") {
        useHudStore.setState({
          label: ev.label,
          trialIndex: ev.trialIndex,
          blockIndex: ev.blockIndex,
        });
      }
      if (ev.type === "TRIAL_COMPLETED") {
        useHudStore.setState({ score: ev.trialScore });
        const sid = sessionIdRef.current;
        if (sid) {
          void apiPostTrial({
            sessionId: sid,
            trialIndex: ev.trialIndex,
            blockIndex: ev.blockIndex,
            label: ev.label,
            targetSide: ev.targetSide,
            outcome: ev.outcome,
            score: ev.scoreDelta,
          }).catch((e: unknown) =>
            setLastError(e instanceof Error ? e.message : "Trial save failed"),
          );
        }
      }
      if (ev.type === "SCORE_CHANGED") {
        useHudStore.setState({ score: ev.totalScore });
      }
      if (ev.type === "TELEMETRY_BATCH_READY") {
        const sid =
          sessionIdRef.current ?? ev.samples[0]?.sessionId ?? null;
        if (sid && ev.samples.length > 0) {
          void apiPostTelemetry(sid, ev.samples)
            .then(() => setLastError(null))
            .catch((e: unknown) => {
              setLastError(
                e instanceof Error ? e.message : "Telemetry failed",
              );
            });
        }
      }
      if (ev.type === "SESSION_COMPLETED") {
        useHudStore.setState({ sessionStatus: "completed" });
        sessionIdRef.current = ev.sessionId;
        void apiFinalizeSession(ev.sessionId, {
          totalScore: ev.totalScore,
          successCount: ev.successCount,
          trialCount: ev.trialCount,
        }).catch((e: unknown) =>
          setLastError(e instanceof Error ? e.message : "Finalize failed"),
        );
      }
      if (ev.type === "ERROR") {
        setLastError(ev.message);
      }
      if (ev.type === "BCI_INPUT_STATUS") {
        const line = `${ev.kind}: ${ev.state}${ev.detail ? ` — ${ev.detail}` : ""}`;
        useHudStore.setState({ bciStatusLine: line });
      }
    });
    return unsub;
  }, [bridge, sessionIdRef, setLastError]);
}
