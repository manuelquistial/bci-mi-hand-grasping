import { PrismaClient } from "@prisma/client";

/**
 * Dev: attach to `globalThis` so Next.js HMR does not instantiate multiple
 * clients. Prod: module singleton is enough (no HMR); we do not use `globalThis`
 * to match common Prisma + Next guidance for serverless-friendly defaults.
 */
const globalForPrisma = globalThis as unknown as { prisma: PrismaClient };

export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({
    log:
      process.env.NODE_ENV === "development"
        ? ["error", "warn"]
        : ["error"],
  });

if (process.env.NODE_ENV !== "production") {
  globalForPrisma.prisma = prisma;
}

/** Optional: enable WAL for local experimentation (safe no-op if unsupported). */
export async function enableSqliteWalMode(): Promise<void> {
  try {
    await prisma.$executeRawUnsafe(`PRAGMA journal_mode=WAL;`);
  } catch {
    /* ignore */
  }
}
