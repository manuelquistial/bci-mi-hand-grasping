import { prisma } from "@/lib/db/prisma";

export async function createSession(data: {
  mode: string;
  startedAt: Date;
  settingsJson: string;
}) {
  return prisma.session.create({
    data: {
      mode: data.mode,
      startedAt: data.startedAt,
      settingsJson: data.settingsJson,
    },
  });
}

export async function listSessions(limit: number, offset: number) {
  return prisma.session.findMany({
    orderBy: { createdAt: "desc" },
    take: limit,
    skip: offset,
  });
}

export async function getSessionById(id: string) {
  return prisma.session.findUnique({
    where: { id },
    include: {
      trials: { orderBy: { trialIndex: "asc" } },
    },
  });
}

export async function updateSessionEnd(
  id: string,
  data: {
    endedAt: Date;
    totalScore: number;
    successRate: number;
  },
) {
  return prisma.session.update({
    where: { id },
    data: {
      endedAt: data.endedAt,
      totalScore: data.totalScore,
      successRate: data.successRate,
    },
  });
}

export async function countSessions(): Promise<number> {
  return prisma.session.count();
}
