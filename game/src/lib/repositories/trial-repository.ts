import { prisma } from "@/lib/db/prisma";

export async function upsertTrial(data: {
  sessionId: string;
  trialIndex: number;
  blockIndex: number;
  label: string;
  targetSide: string;
  phaseTimestampsJson: string;
  outcome: string | null;
  score: number;
  startedAt: Date | null;
  endedAt: Date | null;
}) {
  return prisma.trial.upsert({
    where: {
      sessionId_trialIndex: {
        sessionId: data.sessionId,
        trialIndex: data.trialIndex,
      },
    },
    create: {
      sessionId: data.sessionId,
      trialIndex: data.trialIndex,
      blockIndex: data.blockIndex,
      label: data.label,
      targetSide: data.targetSide,
      phaseTimestampsJson: data.phaseTimestampsJson,
      outcome: data.outcome,
      score: data.score,
      startedAt: data.startedAt,
      endedAt: data.endedAt,
    },
    update: {
      blockIndex: data.blockIndex,
      label: data.label,
      targetSide: data.targetSide,
      phaseTimestampsJson: data.phaseTimestampsJson,
      outcome: data.outcome,
      score: data.score,
      startedAt: data.startedAt,
      endedAt: data.endedAt,
    },
  });
}
