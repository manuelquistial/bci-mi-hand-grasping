import type { GameCommand, GameEvent, GameEventListener } from "@/types/bridge";

export class GameBridge {
  private readonly listeners = new Set<GameEventListener>();
  private readonly commandQueue: GameCommand[] = [];

  subscribe(listener: GameEventListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  emit(event: GameEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (err) {
        console.error("[GameBridge] listener error", err);
      }
    }
  }

  dispatchCommand(command: GameCommand): void {
    this.commandQueue.push(command);
  }

  consumeCommands(): GameCommand[] {
    const next = [...this.commandQueue];
    this.commandQueue.length = 0;
    return next;
  }

  peekCommands(): readonly GameCommand[] {
    return this.commandQueue;
  }
}
