# Bundled reference clips (issue #17)

Human reference pronunciations for languages the OS cannot speak
(esp. Newari — no system SpeechSynthesis voice exists).

## Convention

```
public/reference/<lang>/<exercise-slug>.<webm|mp3|ogg>
```

- `<lang>`: `da`, `en`, `ne`, `new`
- `<exercise-slug>`: lowercase, accents stripped, non-alphanumerics → `-`
  (e.g. "Hello world" → `hello-world.webm`) — see `exerciseSlug()` in
  `src/ui/reference.ts`; a unit test pins the exact mapping.
- Format: prefer **webm/opus** (small), mp3/ogg accepted.
- Content: ONE exercise per clip, native speaker, ~1 s lead-in silence
  trimmed, no background noise.

## Recording checklist (Newari pilot)

- [ ] Record each Newari exercise with a native speaker (16-bit WAV ≥44.1 kHz source is fine)
- [ ] Trim silence, normalise peak to −3 dBFS
- [ ] Export webm/opus (`ffmpeg -i in.wav -c:a libopus -b:a 48k out.webm`)
- [ ] Drop into `public/reference/new/` with the slug name
- [ ] The app picks clips up automatically (HEAD-probed, no code change),
      and uses their F0 as the intonation template for prosody scoring (#16)
