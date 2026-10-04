import { NextResponse } from "next/server";

import { routeErrorResponse } from "@/lib/http/route-errors";
import { countSuccess } from "@/lib/telemetry/aggregate";
import {
  getSessionById,
  updateSessionEnd,
} from "@/lib/repositories/session-repository";
import { patchSessionBodySchema } from "@/lib/validation/session-patch";

type Ctx = { params: Promise<{ id: string }> };

export async function PATCH(req: Request, ctx: Ctx) {
  const { id } = await ctx.params;
  try {
    const json: unknown = await req.json();
    const body = patchSessionBodySchema.parse(json);
    await updateSessionEnd(id, {
      endedAt: new Date(body.endedAt),
      totalScore: body.totalScore,
      successRate: body.successRate,
    });
    return NextResponse.json({ ok: true });
  } catch (e) {
    return routeErrorResponse(e);
  }
}

export async function GET(_req: Request, ctx: Ctx) {
  try {
    const { id } = await ctx.params;
    const session = await getSessionById(id);
    if (!session) {
      return NextResponse.json({ error: "Not found" }, { status: 404 });
    }
    const outcomes = session.trials
      .map((t) => t.outcome)
      .filter((o): o is string => o != null);
    const successCount = countSuccess(outcomes);
    return NextResponse.json({
      session: {
        id: session.id,
        mode: session.mode,
        startedAt: session.startedAt.toISOString(),
        endedAt: session.endedAt?.toISOString() ?? null,
        totalScore: session.totalScore,
        successRate: session.successRate,
        settingsJson: JSON.parse(session.settingsJson) as unknown,
        createdAt: session.createdAt.toISOString(),
      },
      trials: session.trials.map((t) => ({
        id: t.id,
        trialIndex: t.trialIndex,
        blockIndex: t.blockIndex,
        label: t.label,
        targetSide: t.targetSide,
        outcome: t.outcome,
        score: t.score,
      })),
      summary: {
        trialCount: session.trials.length,
        successCount,
      },
    });
  } catch (e) {
    return routeErrorResponse(e);
  }
}
