"use client";

import { useEffect, type MutableRefObject } from "react";

import {
  apiFinalizeSession,
  apiPostTelemetry,
  apiPostTrial,
} from "@/lib/api/client";
import { imageryMarker, pipelineMarker, pipelineOnlineMarker, trialMarker } from "@/lib/pipeline/client";
import type { GameBridge } from "@/game/bridge/game-bridge";
import { useHudStore } from "@/state/hud-store";
import { usePersistenceStore } from "@/state/persistence-store";
import { useSettingsStore } from "@/state/settings-store";
import type { SessionMode } from "@/types/domain";

export function useGameBridgeEvents(
  bridge: GameBridge,
  sessionIdRef: MutableRefObject<string | null>,
) {
  const setLastError = usePersistenceStore((s) => s.setLastError);

  useEffect(() => {
    const modeRef: { current: SessionMode | null } = { current: null };
    const labelRef: { current: string } = { current: "" };
    const imageryOpen: { current: boolean } = { current: false };
    const markImagery = (kind: "imagery_start" | "imagery_end") => {
      const pipeline = useSettingsStore.getState().settings.pipeline;
      if (pipeline.onlineSource !== "cyton") return;
      void pipelineOnlineMarker(pipeline, imageryMarker(labelRef.current, kind)).catch(
        (error: unknown) =>
          setLastError(error instanceof Error ? error.message : "Marker failed"),
      );
    };
    const unsub = bridge.subscribe((ev) => {
      if (ev.type === "SESSION_STARTED") {
        modeRef.current = ev.mode;
        sessionIdRef.current = ev.sessionId;
        useHudStore.setState({
          sessionId: ev.sessionId,
          mode: ev.mode,
          sessionStatus: "running",
          bciStatusLine: null,
        });
        setLastError(null);
      }
      if (ev.type === "SESSION_STOPPED") {
        if (modeRef.current === "online" && imageryOpen.current) {
          imageryOpen.current = false;
          markImagery("imagery_end");
        }
        useHudStore.setState({ sessionStatus: "idle", bciStatusLine: null });
        sessionIdRef.current = null;
        modeRef.current = null;
      }
      if (ev.type === "PHASE_CHANGED") {
        useHudStore.setState({
          phase: ev.phase,
          trialIndex: ev.trialIndex,
          blockIndex: ev.blockIndex,
        });
        if (modeRef.current === "online" && ev.phase === "control") {
          imageryOpen.current = true;
          markImagery("imagery_start");
        } else if (modeRef.current === "online" && imageryOpen.current) {
          imageryOpen.current = false;
          markImagery("imagery_end");
        }
      }
      if (ev.type === "TRIAL_STARTED") {
        labelRef.current = ev.label;
        useHudStore.setState({
          label: ev.label,
          trialIndex: ev.trialIndex,
          blockIndex: ev.blockIndex,
        });
        if (modeRef.current === "offline") {
          const pipeline = useSettingsStore.getState().settings.pipeline;
          void pipelineMarker(pipeline, trialMarker(ev.label, "trial_start")).catch(
            (error: unknown) =>
              setLastError(error instanceof Error ? error.message : "Marker failed"),
          );
        }
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
        if (modeRef.current === "offline") {
          const pipeline = useSettingsStore.getState().settings.pipeline;
          void pipelineMarker(pipeline, trialMarker(ev.label, "trial_end")).catch(
            (error: unknown) =>
              setLastError(error instanceof Error ? error.message : "Marker failed"),
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
