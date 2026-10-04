-- CreateTable
CREATE TABLE "Session" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "mode" TEXT NOT NULL,
    "startedAt" DATETIME NOT NULL,
    "endedAt" DATETIME,
    "totalScore" INTEGER NOT NULL DEFAULT 0,
    "successRate" REAL,
    "settingsJson" TEXT NOT NULL,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- CreateTable
CREATE TABLE "Trial" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "sessionId" TEXT NOT NULL,
    "trialIndex" INTEGER NOT NULL,
    "blockIndex" INTEGER NOT NULL,
    "label" TEXT NOT NULL,
    "targetSide" TEXT NOT NULL,
    "phaseTimestampsJson" TEXT NOT NULL DEFAULT '{}',
    "outcome" TEXT,
    "score" INTEGER NOT NULL DEFAULT 0,
    "startedAt" DATETIME,
    "endedAt" DATETIME,
    CONSTRAINT "Trial_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "Session" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "TelemetrySample" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "sessionId" TEXT NOT NULL,
    "trialId" TEXT,
    "timestamp" REAL NOT NULL,
    "leftValue" REAL NOT NULL,
    "rightValue" REAL NOT NULL,
    "controlSignal" REAL NOT NULL,
    "smoothedSignal" REAL NOT NULL,
    "velocityX" REAL NOT NULL,
    "avatarX" REAL NOT NULL,
    "phase" TEXT NOT NULL,
    CONSTRAINT "TelemetrySample_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "Session" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "TelemetrySample_trialId_fkey" FOREIGN KEY ("trialId") REFERENCES "Trial" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "SessionEvent" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "sessionId" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "payloadJson" TEXT,
    "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "SessionEvent_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "Session" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateIndex
CREATE INDEX "Session_createdAt_idx" ON "Session"("createdAt");

-- CreateIndex
CREATE INDEX "Trial_sessionId_idx" ON "Trial"("sessionId");

-- CreateIndex
CREATE UNIQUE INDEX "Trial_sessionId_trialIndex_key" ON "Trial"("sessionId", "trialIndex");

-- CreateIndex
CREATE INDEX "TelemetrySample_sessionId_timestamp_idx" ON "TelemetrySample"("sessionId", "timestamp");

-- CreateIndex
CREATE INDEX "TelemetrySample_trialId_idx" ON "TelemetrySample"("trialId");

-- CreateIndex
CREATE INDEX "SessionEvent_sessionId_idx" ON "SessionEvent"("sessionId");
