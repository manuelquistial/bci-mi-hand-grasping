import { Prisma } from "@prisma/client";
import { NextResponse } from "next/server";
import { ZodError } from "zod";

export function routeErrorResponse(e: unknown, fallback = "Request failed") {
  if (e instanceof ZodError) {
    return NextResponse.json(
      { error: "Validation failed", issues: e.flatten() },
      { status: 422 },
    );
  }
  if (
    e instanceof Prisma.PrismaClientKnownRequestError ||
    e instanceof Prisma.PrismaClientValidationError
  ) {
    return NextResponse.json(
      { error: "Database error", detail: e instanceof Error ? e.message : "" },
      { status: 500 },
    );
  }
  if (e instanceof Error) {
    return NextResponse.json({ error: e.message }, { status: 400 });
  }
  return NextResponse.json({ error: fallback }, { status: 500 });
}
