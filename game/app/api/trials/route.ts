import { NextResponse } from "next/server";

import { routeErrorResponse } from "@/lib/http/route-errors";
import { upsertTrial } from "@/lib/repositories/trial-repository";
import { postTrialBodySchema } from "@/lib/validation/trials";

export async function POST(req: Request) {
  try {
    const json: unknown = await req.json();
    const body = postTrialBodySchema.parse(json);
    const trial = await upsertTrial({
      sessionId: body.sessionId,
      trialIndex: body.trialIndex,
      blockIndex: body.blockIndex,
      label: body.label,
      targetSide: body.targetSide,
      phaseTimestampsJson: JSON.stringify(body.phaseTimestampsJson ?? {}),
      outcome: body.outcome ?? null,
      score: body.score ?? 0,
      startedAt: body.startedAt ? new Date(body.startedAt) : null,
      endedAt: body.endedAt ? new Date(body.endedAt) : null,
    });
    return NextResponse.json({ id: trial.id, trialIndex: trial.trialIndex });
  } catch (e) {
    return routeErrorResponse(e);
  }
}
