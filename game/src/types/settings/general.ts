export type GeneralSettings = {
  fullscreenPlaceholder: boolean;
  audioPlaceholder: boolean;
};

export const defaultGeneralSettings = (): GeneralSettings => ({
  fullscreenPlaceholder: false,
  audioPlaceholder: false,
});
