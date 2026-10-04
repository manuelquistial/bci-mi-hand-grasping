import { prisma } from "@/lib/db/prisma";

export async function buildSessionExport(sessionId: string) {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    include: {
      trials: { orderBy: { trialIndex: "asc" } },
      telemetrySamples: { orderBy: { timestamp: "asc" } },
    },
  });
  return session;
}

export function toCsv(exportData: NonNullable<Awaited<ReturnType<typeof buildSessionExport>>>): string {
  const lines: string[] = [];
  const telHeader =
    "sample_id,session_id,trial_id,timestamp,left_value,right_value,control_signal,smoothed_signal,velocity_x,avatar_x,phase";
  lines.push(telHeader);
  for (const s of exportData.telemetrySamples) {
    lines.push(
      [
        s.id,
        s.sessionId,
        s.trialId ?? "",
        String(s.timestamp),
        String(s.leftValue),
        String(s.rightValue),
        String(s.controlSignal),
        String(s.smoothedSignal),
        String(s.velocityX),
        String(s.avatarX),
        s.phase,
      ].join(","),
    );
  }
  return lines.join("\n");
}
