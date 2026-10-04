import { z } from "zod";

export const postSessionBodySchema = z.object({
  mode: z.enum(["offline", "online"]),
  settings: z.record(z.string(), z.any()),
  startedAt: z.string().datetime().optional(),
});

export const listSessionsQuerySchema = z.object({
  limit: z.coerce.number().min(1).max(100).optional().default(20),
  offset: z.coerce.number().min(0).optional().default(0),
});

export type PostSessionBody = z.infer<typeof postSessionBodySchema>;
