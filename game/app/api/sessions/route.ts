import { NextResponse } from "next/server";

import { routeErrorResponse } from "@/lib/http/route-errors";
import { createSession, listSessions } from "@/lib/repositories/session-repository";
import { listSessionsQuerySchema, postSessionBodySchema } from "@/lib/validation/sessions";

export async function POST(req: Request) {
  try {
    const json: unknown = await req.json();
    const body = postSessionBodySchema.parse(json);
    const startedAt = body.startedAt ? new Date(body.startedAt) : new Date();
    const session = await createSession({
      mode: body.mode,
      startedAt,
      settingsJson: JSON.stringify(body.settings),
    });
    return NextResponse.json({
      id: session.id,
      mode: session.mode,
      startedAt: session.startedAt.toISOString(),
      settingsJson: JSON.parse(session.settingsJson) as unknown,
    });
  } catch (e) {
    return routeErrorResponse(e);
  }
}

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const q = listSessionsQuerySchema.parse({
      limit: searchParams.get("limit") ?? undefined,
      offset: searchParams.get("offset") ?? undefined,
    });
    const rows = await listSessions(q.limit, q.offset);
    return NextResponse.json({
      sessions: rows.map((s) => ({
        id: s.id,
        mode: s.mode,
        startedAt: s.startedAt.toISOString(),
        endedAt: s.endedAt?.toISOString() ?? null,
        totalScore: s.totalScore,
        successRate: s.successRate,
        createdAt: s.createdAt.toISOString(),
      })),
    });
  } catch (e) {
    return routeErrorResponse(e);
  }
}
