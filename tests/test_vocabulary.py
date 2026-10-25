from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from local_dictation import vocabulary
from local_dictation.stt import server


class _StoreTestCase(unittest.TestCase):
    def _store(self, name: str = "vocabulary.json") -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name) / name

    def _quietly(self) -> redirect_stdout[io.StringIO]:
        return redirect_stdout(io.StringIO())


class VocabularyStoreTests(_StoreTestCase):
    def test_missing_store_loads_empty(self) -> None:
        self.assertEqual(vocabulary.load(self._store()).replacements, ())

    def test_default_location_follows_configuration(self) -> None:
        store = self._store()
        with mock.patch.object(vocabulary.config, "VOCABULARY_PATH", store):
            with self._quietly():
                vocabulary.add_replacement("herder", "herdr")
            self.assertTrue(store.exists())
            self.assertEqual(len(vocabulary.load().replacements), 1)

    def test_add_normalizes_and_persists_private_json(self) -> None:
        store = self._store("nested/vocabulary.json")
        output = io.StringIO()
        with redirect_stdout(output):
            vocabulary.add_replacement("  Herder   Tool ", " herdr tool ", path=store)

        self.assertEqual(
            output.getvalue(), "added replacement 'Herder Tool' -> 'herdr tool'\n"
        )
        self.assertEqual(store.stat().st_mode & 0o777, 0o600)
        self.assertEqual(store.parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual(
            json.loads(store.read_text()),
            {
                "version": vocabulary.SCHEMA_VERSION,
                "replacements": [{"spoken": "Herder Tool", "written": "herdr tool"}],
            },
        )
        self.assertEqual([path.name for path in store.parent.iterdir()], [store.name])

    def test_empty_phrase_or_written_form_is_rejected(self) -> None:
        store = self._store()
        for spoken, written in (("  ", "x"), ("x", "  ")):
            with (
                self.subTest(spoken=spoken, written=written),
                self.assertRaisesRegex(vocabulary.VocabularyError, "required"),
            ):
                vocabulary.add_replacement(spoken, written, path=store)
        self.assertFalse(store.exists())

    def test_conflicting_add_is_rejected_without_force(self) -> None:
        store = self._store()
        with self._quietly():
            vocabulary.add_replacement("cursa", "Cursor", path=store)
        with self.assertRaisesRegex(vocabulary.VocabularyError, "--force"):
            vocabulary.add_replacement("CURSA", "cursor", path=store)

        output = io.StringIO()
        with redirect_stdout(output):
            vocabulary.add_replacement("CURSA", "cursor", force=True, path=store)

        self.assertIn("updated replacement", output.getvalue())
        self.assertEqual(
            vocabulary.load(store).replacements,
            (vocabulary.Replacement("CURSA", "cursor"),),
        )

    def test_identical_add_is_idempotent(self) -> None:
        store = self._store()
        with self._quietly():
            vocabulary.add_replacement("kubernetes", "Kubernetes", path=store)
            vocabulary.add_replacement("kubernetes", "Kubernetes", path=store)
        self.assertEqual(len(vocabulary.load(store).replacements), 1)

    def test_remove_deletes_case_insensitively_and_reports_missing(self) -> None:
        store = self._store()
        with self._quietly():
            vocabulary.add_replacement("herder", "herdr", path=store)
            vocabulary.add_replacement("cursa", "Cursor", path=store)
            vocabulary.remove_replacement("  HERDER ", path=store)

        self.assertEqual(
            vocabulary.load(store).replacements,
            (vocabulary.Replacement("cursa", "Cursor"),),
        )
        with self.assertRaisesRegex(vocabulary.VocabularyError, "no replacement"):
            vocabulary.remove_replacement("absent", path=store)

    def test_list_reports_entries_sorted_and_emptiness(self) -> None:
        store = self._store()
        empty = io.StringIO()
        with redirect_stdout(empty):
            vocabulary.list_entries(path=store)
        self.assertEqual(empty.getvalue(), "replacements:\n  (none)\n")

        with self._quietly():
            vocabulary.add_replacement("zed", "Zed", path=store)
            vocabulary.add_replacement("herder", "herdr", path=store)
        populated = io.StringIO()
        with redirect_stdout(populated):
            vocabulary.list_entries(path=store)

        self.assertEqual(
            populated.getvalue(),
            "replacements:\n  'herder' -> 'herdr'\n  'zed' -> 'Zed'\n",
        )


class VocabularyValidationTests(_StoreTestCase):
    def _load(self, document: object) -> vocabulary.Vocabulary:
        store = self._store()
        store.write_text(json.dumps(document))
        return vocabulary.load(store)

    def test_malformed_json_is_rejected(self) -> None:
        store = self._store()
        store.write_text("{not json")
        with self.assertRaisesRegex(vocabulary.VocabularyError, "not valid JSON"):
            vocabulary.load(store)

    def test_invalid_documents_are_rejected(self) -> None:
        cases: list[tuple[str, object]] = [
            ("store .* must be an object", []),
            ("unsupported vocabulary schema version", {"version": 99}),
            ("must be a list", {"replacements": {}}),
            ("each replacement must be an object", {"replacements": ["herder"]}),
            ("must be strings", {"replacements": [{"spoken": "x", "written": 1}]}),
            ("must not be empty", {"replacements": [{"spoken": " ", "written": "x"}]}),
        ]
        for message, document in cases:
            with (
                self.subTest(message),
                self.assertRaisesRegex(vocabulary.VocabularyError, message),
            ):
                self._load(document)

    def test_ambiguous_stored_file_is_rejected_on_load(self) -> None:
        with self.assertRaisesRegex(vocabulary.VocabularyError, "ambiguous"):
            self._load(
                {
                    "replacements": [
                        {"spoken": "x", "written": "one"},
                        {"spoken": "X", "written": "two"},
                    ]
                }
            )

    def test_duplicate_identical_entries_collapse(self) -> None:
        loaded = self._load(
            {
                "replacements": [
                    {"spoken": "x", "written": "one"},
                    {"spoken": "x", "written": "one"},
                ]
            }
        )
        self.assertEqual(loaded.replacements, (vocabulary.Replacement("x", "one"),))

    def test_local_voice_harness_documents_keep_only_replacements(self) -> None:
        loaded = self._load(
            {
                "version": 2,
                "replacements": [{"spoken": "herder", "written": "herdr"}],
                "aliases": [{"phrase": "the repo", "target": "owner/repo"}],
                "pronunciations": [{"written": "Herdr", "spoken": "herder"}],
            }
        )

        self.assertEqual(loaded.version, vocabulary.SCHEMA_VERSION)
        self.assertEqual(
            loaded.replacements, (vocabulary.Replacement("herder", "herdr"),)
        )
        self.assertEqual(set(loaded.to_document()), {"version", "replacements"})


class VocabularyPortabilityTests(_StoreTestCase):
    def test_export_and_import_round_trip(self) -> None:
        store = self._store("original.json")
        backup = self._store("backup.json")
        exported = io.StringIO()
        with redirect_stdout(exported):
            vocabulary.add_replacement("herder", "herdr", path=store)
            vocabulary.export_entries(backup, path=store)
        self.assertIn(f"exported vocabulary to {backup}", exported.getvalue())

        printed = io.StringIO()
        with redirect_stdout(printed):
            vocabulary.export_entries(path=store)
        self.assertEqual(
            json.loads(printed.getvalue()), vocabulary.load(store).to_document()
        )

        destination = self._store("imported.json")
        imported = io.StringIO()
        with redirect_stdout(imported):
            vocabulary.import_entries(backup, path=destination)

        self.assertIn("imported 1 replacement(s)", imported.getvalue())
        self.assertEqual(vocabulary.load(destination), vocabulary.load(store))

    def test_import_merge_prefers_incoming(self) -> None:
        store = self._store()
        backup = self._store("backup.json")
        with self._quietly():
            vocabulary.add_replacement("keep", "kept", path=store)
            vocabulary.add_replacement("herder", "old", path=store)
            vocabulary.add_replacement("herder", "herdr", path=backup)
            vocabulary.import_entries(backup, path=store)

        self.assertEqual(
            {item.spoken: item.written for item in vocabulary.load(store).replacements},
            {"keep": "kept", "herder": "herdr"},
        )

    def test_import_replace_overwrites_existing(self) -> None:
        store = self._store()
        backup = self._store("backup.json")
        output = io.StringIO()
        with redirect_stdout(output):
            vocabulary.add_replacement("keep", "kept", path=store)
            vocabulary.add_replacement("new", "fresh", path=backup)
            vocabulary.import_entries(backup, replace=True, path=store)

        self.assertIn("replaced vocabulary with 1 replacement(s)", output.getvalue())
        self.assertEqual(
            vocabulary.load(store).replacements,
            (vocabulary.Replacement("new", "fresh"),),
        )

    def test_import_rejects_unreadable_or_invalid_sources(self) -> None:
        store = self._store()
        invalid = self._store("invalid.json")
        invalid.write_text("{broken")
        not_object = self._store("list.json")
        not_object.write_text("[]")
        cases = {
            "could not read": self._store("missing.json"),
            "is not valid JSON": invalid,
            "must be a JSON object": not_object,
        }
        for message, source in cases.items():
            with (
                self.subTest(message),
                self.assertRaisesRegex(vocabulary.VocabularyError, message),
            ):
                vocabulary.import_entries(source, path=store)
        self.assertFalse(store.exists())


class ReplacementMatchingTests(unittest.TestCase):
    def test_whole_phrase_matches_ignore_case_and_whitespace(self) -> None:
        replacement = vocabulary.Replacement("herder tool", "herdr")
        self.assertEqual(
            replacement.apply("Run the HERDER   tool, not the herdertool."),
            "Run the herdr, not the herdertool.",
        )

    def test_written_form_is_inserted_literally(self) -> None:
        replacement = vocabulary.Replacement("path", r"C:\new\1")
        self.assertEqual(replacement.apply("a path here"), r"a C:\new\1 here")


class SpeechToTextCorrectionTests(_StoreTestCase):
    def test_missing_store_preserves_static_replacements(self) -> None:
        store = self._store()
        with mock.patch.object(server.config, "VOCABULARY_PATH", store):
            self.assertEqual(server.normalize("use curser now"), "use Cursor now")

    def test_user_replacement_overrides_static_and_defaults(self) -> None:
        store = self._store()
        vocabulary.save(
            vocabulary.Vocabulary(
                replacements=(
                    vocabulary.Replacement("herder", "Herder-Custom"),
                    vocabulary.Replacement("kubernetes", "Kubernetes"),
                )
            ),
            store,
        )
        with mock.patch.object(server.config, "VOCABULARY_PATH", store):
            self.assertEqual(
                server.normalize("the herder runs kubernetes and curser"),
                "the Herder-Custom runs Kubernetes and Cursor",
            )

    def test_broken_store_falls_back_to_static_replacements(self) -> None:
        store = self._store()
        store.write_text("{broken")
        with mock.patch.object(server.config, "VOCABULARY_PATH", store):
            self.assertEqual(server.normalize("use curser"), "use Cursor")


if __name__ == "__main__":
    unittest.main()
