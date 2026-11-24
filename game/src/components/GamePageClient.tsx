"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";

import { GameBridge } from "@/game/bridge/game-bridge";
import type { CollectorStartPayload } from "@/game/scenes/collector-session-types";
import { useGameBridgeEvents } from "@/hooks/useGameBridgeEvents";
import { pipelinePreModelFilter, type FilterPreview } from "@/lib/pipeline/client";
import { enterFullscreen, exitFullscreen } from "@/lib/session/fullscreen";
import {
  startSessionRuntime,
  stopSessionPipeline,
} from "@/lib/session/session-lifecycle";
import { useHudStore } from "@/state/hud-store";
import { usePersistenceStore } from "@/state/persistence-store";
import { useSettingsStore } from "@/state/settings-store";

import { FilterCharts } from "./FilterCharts";
import { GameControls } from "./GameControls";
import { SettingsPanel, type SetupScreen } from "./SettingsPanel";
import styles from "./game-page.module.css";

const PhaserGameContainer = dynamic(
  () =>
    import("@/components/PhaserGameContainer").then(
      (m) => m.PhaserGameContainer,
    ),
  { ssr: false },
);

type View = "setup" | "session" | "charts";

export function GamePageClient({ initialStep }: { initialStep: SetupScreen }) {
  const bridge = useMemo(() => new GameBridge(), []);
  const sessionIdRef = useRef<string | null>(null);
  const viewRef = useRef<View>("setup");
  /** Serialize pipeline stop/start so Space→Start cannot race hub teardown. */
  const pipelineGate = useRef(Promise.resolve<void>(undefined));
  useGameBridgeEvents(bridge, sessionIdRef);

  const settings = useSettingsStore((s) => s.settings);
  const setMode = useSettingsStore((s) => s.setMode);
  const setNested = useSettingsStore((s) => s.setNested);
  const lastError = usePersistenceStore((s) => s.lastError);
  const setLastError = usePersistenceStore((s) => s.setLastError);
  const sessionStatus = useHudStore((s) => s.sessionStatus);
  const [view, setView] = useState<View>("setup");
  const [starting, setStarting] = useState(false);
  const [preview, setPreview] = useState<FilterPreview | null>(null);
  const [startPayload, setStartPayload] = useState<CollectorStartPayload | null>(
    null,
  );

  useEffect(() => {
    if (view !== "charts") return;
    window.scrollTo(0, 0);
  }, [view]);

  useEffect(() => {
    if (initialStep === "offline") setMode("offline");
    if (initialStep === "simulator" || initialStep === "real") {
      setMode("online");
      setNested("pipeline", {
        onlineSource: initialStep === "real" ? "cyton" : "simulator",
      });
    }
  }, [initialStep, setMode, setNested]);

  const showSession = (next: View) => {
    viewRef.current = next;
    setView(next);
  };

  useEffect(() => {
    if (sessionStatus !== "running") return;
    bridge.dispatchCommand({
      type: "UPDATE_SETTINGS",
      payload: { settings },
    });
  }, [bridge, sessionStatus, settings]);

  const leaveSession = (opts: { stopPipeline: boolean; stopPhaser: boolean }) => {
    if (viewRef.current !== "session") return;
    setStartPayload(null);
    showSession("setup");
    if (opts.stopPhaser) {
      bridge.dispatchCommand({ type: "STOP_SESSION" });
    }
    if (opts.stopPipeline) {
      const st = useSettingsStore.getState().settings;
      const md = useSettingsStore.getState().mode;
      const stopping = stopSessionPipeline(md, st)
        .then(() => undefined)
        .catch((error: unknown) => {
          setLastError(
            error instanceof Error ? error.message : "Could not stop the pipeline",
          );
        });
      pipelineGate.current = pipelineGate.current.then(() => stopping);
    }
    void exitFullscreen();
    useHudStore.setState({ sessionStatus: "idle", bciStatusLine: null });
  };

  // Simulator/online finished all trials → leave Phaser and return to setup.
  useEffect(() => {
    if (sessionStatus !== "completed") return;
    if (viewRef.current !== "session") return;
    leaveSession({ stopPipeline: true, stopPhaser: false });
    // leaveSession is stable enough via viewRef; avoid re-binding loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionStatus]);

  const stop = () => {
    leaveSession({ stopPipeline: true, stopPhaser: true });
  };
  const stopRef = useRef(stop);
  stopRef.current = stop;

  useEffect(() => {
    if (view !== "session") return;
    const onKey = (event: KeyboardEvent) => {
      if (event.code !== "Space") return;
      if (event.repeat) return;
      const target = event.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement
      ) {
        return;
      }
      event.preventDefault();
      stopRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [view]);

  const start = async () => {
    const st = useSettingsStore.getState().settings;
    const md = useSettingsStore.getState().mode;
    setLastError(null);
    setStarting(true);
    try {
      // Wait out any in-flight stop from Space / completed session.
      await pipelineGate.current;
      if (initialStep === "pre-model") {
        const next = await pipelinePreModelFilter(st);
        setPreview(next);
        viewRef.current = "charts";
        setView("charts");
        return;
      }
      await enterFullscreen();
      const started = await startSessionRuntime(md, st);
      sessionIdRef.current = started.sessionId;
      setStartPayload(started.payload);
      showSession("session");
      // START_SESSION is dispatched by PhaserGameContainer after BootScene is ready.
    } catch (error: unknown) {
      setStartPayload(null);
      showSession("setup");
      setLastError(error instanceof Error ? error.message : "Could not start session");
      void exitFullscreen();
    } finally {
      setStarting(false);
    }
  };

  if (view === "charts" && preview) {
    return (
      <div className={styles.chartsView}>
        <button
          type="button"
          className={styles.back}
          onClick={() => {
            viewRef.current = "setup";
            setView("setup");
          }}
        >
          Back
        </button>
        <FilterCharts preview={preview} />
      </div>
    );
  }

  if (view === "session" && startPayload) {
    return (
      <div className={styles.session}>
        <div className={styles.stage}>
          <PhaserGameContainer bridge={bridge} startPayload={startPayload} />
        </div>
        <p className={styles.stopHint}>Press space to stop</p>
      </div>
    );
  }

  const openMonitor = () => {
    const pipeline = useSettingsStore.getState().settings.pipeline;
    const params = new URLSearchParams({
      baseUrl: pipeline.baseUrl || "http://127.0.0.1:8765",
      controlPort: String(pipeline.controlWsPort || 8090),
    });
    window.open(
      `/monitor?${params.toString()}`,
      "bci-online-monitor",
      "noopener,noreferrer",
    );
  };

  return (
    <div className={styles.setup}>
      <SettingsPanel screen={initialStep} />
      <GameControls
        onStart={start}
        starting={starting}
        label={initialStep === "pre-model" ? "Start pre model" : "Start session"}
        showMonitor={initialStep === "simulator" || initialStep === "real"}
        onOpenMonitor={openMonitor}
      />
      {lastError ? (
        <p className={styles.error} role="alert">
          {lastError}
        </p>
      ) : null}
    </div>
  );
}
