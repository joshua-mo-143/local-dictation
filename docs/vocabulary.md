# Vocabulary

The vocabulary store holds your own transcript corrections: project names,
developer tools, acronyms, people, and words the recognizer keeps getting
wrong. You edit it only through `dictate vocabulary`; dictation never learns a
correction on its own, and the store never leaves the machine.

## Storage

The store is a single JSON document at
`$XDG_CONFIG_HOME/local-dictation/vocabulary.json` (by default
`~/.config/local-dictation/vocabulary.json`). It's written atomically with
owner-only permissions (`0600`) and sorted keys, so backups and diffs stay
stable.

Each **replacement** rewrites the recognized text (`spoken`) to the form you
want typed (`written`):

```json
{
  "replacements": [
    {
      "spoken": "herder",
      "written": "herdr"
    }
  ],
  "version": 3
}
```

## Matching and precedence

- Spoken phrases are trimmed, have runs of whitespace collapsed, and are compared
  case-insensitively. Written forms keep their case.
- Matching is case-insensitive and whitespace-flexible, and only fires on whole
  phrases, never inside a longer word.
- Vocabulary replacements win over `dictation.replacements` from
  [configuration](configuration.md), which win over the built-in defaults.
- Each spoken phrase maps to one written form. `add` rejects a phrase that
  already maps to something else unless you pass `--force`, and a stored file
  that maps the same phrase two different ways is rejected as ambiguous when
  it's loaded.

The service reads the store for each transcription, so changes apply to the
next transcript without a restart. If the file is missing or broken, the service
keeps transcribing with the configured and built-in replacements.

## Commands

```fish
dictate vocabulary list
dictate vocabulary add herder herdr
dictate vocabulary add "cursor rules" "Cursor rules" --force
dictate vocabulary remove herder
dictate vocabulary export --output vocabulary-backup.json
dictate vocabulary import vocabulary-backup.json
dictate vocabulary import vocabulary-backup.json --replace
```

`export` without `--output` prints the document to stdout. `import` merges by
default, with incoming entries winning on conflict, or replaces the whole store
with `--replace`. To delete the store entirely, remove the file.

## Importing from local-voice-harness

Vocabulary exports from local-voice-harness (schema versions 1 to 3) import
directly. Their replacements carry over, while aliases and pronunciations, which
only the voice assistant used, are ignored:

```fish
dictate vocabulary import ~/.config/voice-harness/vocabulary.json
```
