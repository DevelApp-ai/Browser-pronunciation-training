/**
 * i18n-ready UI strings — issue #9.
 * UI must eventually support da/en/ne/Newari contexts.
 */
export const STRINGS = {
  en: {
    record: "Record",
    stop: "Stop",
    play: "Play",
    playReference: "Hear reference",
    slow: "Slow (0.7×)",
    retry: "Try again",
    listening: "Listening…",
    scoring: "Scoring…",
    overall: "Overall score",
    scoreNone: "Record yourself to get feedback",
    youSaid: "you said",
    target: "target",
    micDenied: "Microphone access was denied.",
    noMic: "No microphone found.",
    micUnsupported: "This browser does not support audio capture.",
  },
  da: {
    record: "Optag",
    stop: "Stop",
    play: "Afspil",
    playReference: "Hør reference",
    slow: "Langsomt (0,7×)",
    retry: "Prøv igen",
    listening: "Lytter…",
    scoring: "Vurderer…",
    overall: "Samlet score",
    scoreNone: "Optag dig selv for at få feedback",
    youSaid: "du sagde",
    target: "mål",
    micDenied: "Mikrofonadgang blev nægtet.",
    noMic: "Ingen mikrofon fundet.",
    micUnsupported: "Denne browser understøtter ikke lydoptagelse.",
  },
} as const;

export type Lang = keyof typeof STRINGS;
export const t = (lang: Lang) => STRINGS[lang] ?? STRINGS.en;
