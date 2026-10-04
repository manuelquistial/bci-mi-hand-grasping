import { z } from "zod";

const sampleSchema = z.object({
  trialId: z.string().nullable(),
  timestamp: z.number(),
  leftValue: z.number(),
  rightValue: z.number(),
  controlSignal: z.number(),
  smoothedSignal: z.number(),
  velocityX: z.number(),
  avatarX: z.number(),
  phase: z.string(),
});

export const postTelemetryBodySchema = z.object({
  sessionId: z.string().min(1),
  samples: z.array(sampleSchema).min(1).max(500),
});

export type PostTelemetryBody = z.infer<typeof postTelemetryBodySchema>;
