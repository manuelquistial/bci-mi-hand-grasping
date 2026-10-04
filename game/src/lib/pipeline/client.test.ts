import { describe, expect, it, vi } from "vitest";

import { defaultSessionSettings } from "@/types/settings";

import { pipelineRecordStart, trialMarker } from "./client";

describe("pipeline client", () => {
  it("posts the session id to the record endpoint", async () => {
    const urls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        urls.push(String(url));
        return new Response(JSON.stringify({ path: "/tmp/sess" }), { status: 200 });
      }),
    );
    const settings = defaultSessionSettings().pipeline;
    await pipelineRecordStart(settings, "sess_1");
    expect(urls).toEqual(["http://127.0.0.1:8765/v1/record/start"]);
    vi.unstubAllGlobals();
  });

  it("maps left and right onto T1 and T2", () => {
    expect(trialMarker("left", "trial_start")).toEqual({
      kind: "trial_start",
      code: "T1",
      label: "left",
    });
    expect(trialMarker("right", "trial_end").code).toBe("T2");
  });
});
