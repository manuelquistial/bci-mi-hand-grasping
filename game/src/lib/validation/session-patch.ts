import { z } from "zod";

export const patchSessionBodySchema = z.object({
  endedAt: z.string().datetime(),
  totalScore: z.number().int(),
  successRate: z.number().min(0).max(1),
});
