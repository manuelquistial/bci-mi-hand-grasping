import { NextResponse } from "next/server";

import { routeErrorResponse } from "@/lib/http/route-errors";
import { createTelemetryBatch } from "@/lib/repositories/telemetry-repository";
import { postTelemetryBodySchema } from "@/lib/validation/telemetry";

export async function POST(req: Request) {
  try {
    const json: unknown = await req.json();
    const body = postTelemetryBodySchema.parse(json);
    await createTelemetryBatch(body.sessionId, body.samples);
    return NextResponse.json({ ok: true, count: body.samples.length });
  } catch (e) {
    return routeErrorResponse(e);
  }
}
