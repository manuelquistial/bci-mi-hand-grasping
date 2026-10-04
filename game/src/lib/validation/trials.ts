import { z } from "zod";

export const postTrialBodySchema = z.object({
  sessionId: z.string().min(1),
  trialIndex: z.number().int().min(0),
  blockIndex: z.number().int().min(0),
  label: z.string(),
  targetSide: z.string(),
  phaseTimestampsJson: z.record(z.string(), z.number()).optional(),
  outcome: z.string().nullable().optional(),
  score: z.number().int().optional(),
  startedAt: z.string().datetime().nullable().optional(),
  endedAt: z.string().datetime().nullable().optional(),
});

export type PostTrialBody = z.infer<typeof postTrialBodySchema>;
