"use client";

import { useEffect, useState } from "react";

import {
  connectObserveSocket,
  fetchOnlineStatus,
  observeWsUrlFromControl,
  type PipelineStageMessage,
  type StreamFrameMessage,
} from "@/lib/pipeline/observe";
import { useSettingsStore } from "@/state/settings-store";

export type SocketStatus = "connecting" | "open" | "closed" | "error";

export type OnlineMeta = {
  source?: string | null;
  samples?: number;
  controlSamples?: number | null;
  registry?: string | null;
  error?: string | null;
};

function readMonitorQuery(): { baseUrl: string; controlPort: number } {
  if (typeof window === "undefined") {
    return { baseUrl: "http://127.0.0.1:8765", controlPort: 8090 };
  }
  const params = new URLSearchParams(window.location.search);
  const fromStore = useSettingsStore.getState().settings.pipeline;
  const baseUrl =
    params.get("baseUrl") || fromStore.baseUrl || "http://127.0.0.1:8765";
  const controlPort = Number(
    params.get("controlPort") || fromStore.controlWsPort || 8090,
  );
  return {
    baseUrl,
    controlPort: Number.isFinite(controlPort) ? controlPort : 8090,
  };
}

/** Poll online status and stream the observe socket for the monitor page. */
export function useObserveSession() {
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8765");
  const [observeUrl, setObserveUrl] = useState("ws://127.0.0.1:8091");
  const [socketStatus, setSocketStatus] = useState<SocketStatus>("connecting");
  const [stage, setStage] = useState<PipelineStageMessage | null>(null);
  const [stream, setStream] = useState<StreamFrameMessage | null>(null);
  const [onlineMeta, setOnlineMeta] = useState<OnlineMeta>({});

  useEffect(() => {
    const query = readMonitorQuery();
    setBaseUrl(query.baseUrl);
    setObserveUrl(`ws://127.0.0.1:${query.controlPort + 1}`);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const online = await fetchOnlineStatus(baseUrl);
        if (cancelled) return;
        if (!online) {
          setOnlineMeta({});
          return;
        }
        setOnlineMeta({
          source: online.source,
          samples: online.samples_seen,
          controlSamples: online.control_samples,
          registry: online.registry,
          error: online.error,
        });
        if (online.observe_ws_url) {
          setObserveUrl(online.observe_ws_url);
        } else if (online.control_ws_url) {
          setObserveUrl(observeWsUrlFromControl(online.control_ws_url));
        }
        if (online.last_pipeline_stage) {
          setStage((prev) => prev ?? online.last_pipeline_stage ?? null);
        }
        if (online.last_stream_frame) {
          setStream((prev) => prev ?? online.last_stream_frame ?? null);
        }
      } catch {
        if (!cancelled) {
          setOnlineMeta((prev) => ({ ...prev, error: "status unreachable" }));
        }
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 2000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [baseUrl]);

  useEffect(() => {
    const connection = connectObserveSocket(observeUrl, {
      onMessage: (message) => {
        if (message.type === "pipeline_stage") setStage(message);
        else setStream(message);
      },
      onStatus: setSocketStatus,
    });
    return () => connection.close();
  }, [observeUrl]);

  return { observeUrl, socketStatus, stage, stream, onlineMeta };
}
