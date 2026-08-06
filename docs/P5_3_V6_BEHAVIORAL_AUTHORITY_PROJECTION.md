# P5.3 — Contract v6 Behavioral Authority Projection

P5.3 admits only `native_v3_student_v6` through Contract's generic public
admission entry point with `strict=True`.  It has no v5 or directory-only
fallback.

`open_native_v3_v6_behavioral_projection()` retains no artifact path, locator,
archive-member name, or temporary path.  Its language projection carries the
Contract binding digest, tokenizer and vocabulary identity declarations, and
explicit EOS/PAD IDs; equal EOS/PAD IDs are retained as an intentional alias.
Before projection, Student closes the exact seven-role v6 authority set:
`authority_reference`, `corridor_assignment`, `corridor_mode_table`,
`example_registry`, `selected_exemplar_payload`, `selected_passport_index`,
and `target_shard`, each at its required `/default` resource ID. Missing,
additional, duplicate, or substituted authority entries fail closed;
`delivery_receipt` remains outside that authority set.

Payload access is exclusively through Contract's public v6 APIs.  Both
`target_shard/default` and `corridor_assignment/default` are opened as whole
multipart resources, then every declared component is decoded with
`allow_pickle=False`, checked against Contract-projected dtype/shape/axes, and
copied into read-only arrays.  Example/passport JSONL and selected M7 records
are fully drained through their dedicated openers before becoming immutable
Student values.  The M7 verification state must be `fully_verified`.
The two admitted whole JSON authorities, `authority_reference/default` and
`corridor_mode_table/default`, are opened only through the generic v6 byte
opener, decoded as objects, and retained with their public resource metadata
and identities.  No multipart convenience or component opener is used for
either JSON resource.

Against Tome `8508b1351d0ed8d6a3a14049e4d6f8a849c33cf1`'s unchanged v6 fixture
tree, strict public checks passed for both `student/` and `student.tgz`:

- `canonical_binding_digest` and the v6 `language_binding_digest` were both
  `sha256:9419614278eebaddd4d016b884afe21f4ab83e749ec04be3bb597d212f1d02cf`.
- Complete language descriptors, multipart component metadata, component
  bytes, semantic identities, and enclosing multipart identities matched
  across the directory and archive transports.
- Contract rejected wrong/historical profiles, non-strict language or
  multipart projection, tampering, and unsafe archives before Student received
  a model value.

This checkpoint claims admission and immutable neutral projection only.  It
does not create a split, batch, loss, optimizer, pass schedule, checkpoint, or
evaluation result.
