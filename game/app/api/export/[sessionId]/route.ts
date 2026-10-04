import { NextResponse } from "next/server";

import { routeErrorResponse } from "@/lib/http/route-errors";
import { buildSessionExport, toCsv } from "@/lib/services/export-service";

type Ctx = { params: Promise<{ sessionId: string }> };

export async function GET(req: Request, ctx: Ctx) {
  try {
    const { sessionId } = await ctx.params;
    const { searchParams } = new URL(req.url);
    const format = searchParams.get("format") ?? "json";

    const data = await buildSessionExport(sessionId);
    if (!data) {
      return NextResponse.json({ error: "Not found" }, { status: 404 });
    }

    if (format === "csv") {
      const csv = toCsv(data);
      return new NextResponse(csv, {
        status: 200,
        headers: {
          "Content-Type": "text/csv; charset=utf-8",
          "Content-Disposition": `attachment; filename="session-${sessionId}.csv"`,
        },
      });
    }

    return NextResponse.json(data);
  } catch (e) {
    return routeErrorResponse(e);
  }
}
