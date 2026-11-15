export async function enterFullscreen(): Promise<void> {
  if (document.fullscreenElement) return;
  try {
    await document.documentElement.requestFullscreen();
  } catch {
    // The session view still covers the window if the browser refuses.
  }
}

export async function exitFullscreen(): Promise<void> {
  if (!document.fullscreenElement) return;
  try {
    await document.exitFullscreen();
  } catch {
    // Already left fullscreen.
  }
}
