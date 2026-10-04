import { prisma } from "@/lib/db/prisma";

const CHUNK = 100;

export async function createTelemetryBatch(
  sessionId: string,
  rows: {
    trialId: string | null;
    timestamp: number;
    leftValue: number;
    rightValue: number;
    controlSignal: number;
    smoothedSignal: number;
    velocityX: number;
    avatarX: number;
    phase: string;
  }[],
) {
  for (let i = 0; i < rows.length; i += CHUNK) {
    const slice = rows.slice(i, i + CHUNK);
    await prisma.telemetrySample.createMany({
      data: slice.map((r) => ({
        sessionId,
        trialId: r.trialId,
        timestamp: r.timestamp,
        leftValue: r.leftValue,
        rightValue: r.rightValue,
        controlSignal: r.controlSignal,
        smoothedSignal: r.smoothedSignal,
        velocityX: r.velocityX,
        avatarX: r.avatarX,
        phase: r.phase,
      })),
    });
  }
}

export async function listTelemetryForSession(sessionId: string) {
  return prisma.telemetrySample.findMany({
    where: { sessionId },
    orderBy: { timestamp: "asc" },
  });
}
