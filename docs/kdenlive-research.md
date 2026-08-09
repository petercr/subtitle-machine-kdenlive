# Kdenlive project export research

Status: Milestone 0 research record
Updated: 2026-08-08
Target environment: Kdenlive 26.04.3, MLT/melt 7.40.0, Windows 11

## Decision

The core application will generate Kdenlive project files directly through a
small adapter. It will not depend on GUI automation or a running Kdenlive
process.

The first export target is a current-generation Kdenlive project containing:

1. the source video on a timeline;
2. a sibling SRT file used by the project's subtitle filter; and
3. enough profile and timeline metadata for Kdenlive to open the project with
   the source media synchronized.

ASS remains the preferred asset for FFmpeg burned-in previews. For an editable
Kdenlive subtitle track, SRT is the safer first interchange format because the
current Kdenlive file format represents subtitle tracks around an
`avfilter.subtitles` filter backed by an SRT file.

## What the project format requires

Kdenlive project files use an XML document based on MLT. The project stores
media references and editing metadata; it generally does not embed the media
itself. This makes project generation practical, but it also means that the
exporter must be deliberate about absolute paths, relative paths, and the
project working directory.

The current format documentation describes generation 5 projects with
document version 1.1. The important high-level structure is:

```text
Kdenlive document
├── profile
├── producers              source clips and generated assets
├── playlists              one or more timeline tracks
├── main_bin               project-bin entries
└── tractors               multitrack timeline compositions
```

The active timeline is represented by the final tractor wrapper. A subtitle
track is represented as a locked/fake track with an MLT
`avfilter.subtitles` filter whose `av.filename` points to an SRT file. The SRT
must therefore be delivered beside the project and referenced with a path
that Kdenlive can resolve.

Kdenlive may rewrite or upgrade the document when it is opened. Newer project
files are not guaranteed to be backward compatible with older Kdenlive
versions, so the adapter must record the target format generation and should
not claim broad version portability.

Primary references:

- [Kdenlive project files manual](https://docs.kdenlive.org/en/project_and_asset_management/file_management/project_files.html)
- [Kdenlive file format development notes](https://github.com/KDE/kdenlive/blob/master/dev-docs/fileformat.md)
- [MLT XML authoring documentation](https://mltframework.org/docs/mltxml/)

## Reuse and license review

| Project | Relevant material | License | Decision |
| --- | --- | --- | --- |
| [Kdenlive](https://github.com/KDE/kdenlive) | The authoritative application and format implementation | GPL-3.0 | Use its public format documentation as the compatibility reference; do not copy application internals into the core. |
| [MLT](https://github.com/mltframework/mlt) | XML model, producers, playlists, tractors, filters, and renderer | LGPL-2.1 | Use the documented MLT model and local `melt` for validation; keep the exporter independent of MLT Python bindings. |
| [kdenlive-automation](https://github.com/IO-AtelierTech/kdenlive-automation) | Python API/MCP using JSON-RPC over WebSocket | MIT | Reference only. It requires a running Kdenlive fork with a WebSocket RPC server, which conflicts with the local deterministic core boundary. |
| [CLI-Anything](https://github.com/HKUDS/CLI-Anything) | Generated Kdenlive command-line harness using MLT XML and `melt` | Apache-2.0 | Inspect as an external implementation reference during the spike. Do not copy code until the exact source, provenance, and compatibility behavior have been reviewed. |

The exporter will initially use Python's standard-library XML tooling and
explicitly authored project structures. This keeps the dependency surface
small and makes the generated document inspectable. If code is later reused,
its source file, commit, license, and required attribution will be recorded in
this document before it is added.

## Options considered

### Direct `.kdenlive` generation — selected

Advantages:

- works without launching Kdenlive;
- deterministic and testable from Python;
- fits the existing service/CLI architecture;
- preserves the option to validate or render with local Kdenlive/MLT tools.

Costs:

- the XML format is an application format rather than a stable public API;
- paths and format generations need explicit handling;
- opening a generated project in Kdenlive is required for compatibility
  acceptance.

### MLT-only project generation

MLT XML is a useful foundation and `melt` can render MLT-compatible project
structures. However, Kdenlive-specific project-bin entries, metadata, subtitle
track behavior, and document versioning still need to be represented. MLT is
therefore a validation/rendering tool and format foundation, not a substitute
for a Kdenlive document adapter.

### WebSocket or GUI automation — deferred

The researched automation project is useful for a future optional live-editor
integration, but it requires a running application and a special Kdenlive fork.
Mouse/keyboard automation would be more fragile still. Neither belongs in the
first exporter implementation.

## Proposed adapter boundary

The rest of the application should depend on a project-level interface rather
than Kdenlive XML details:

```python
class ProjectAdapter:
    def create_project(self, source, subtitles, metadata): ...
    def add_video(self, project, source, metadata): ...
    def add_subtitles(self, project, srt_path, metadata): ...
    def save(self, project, destination): ...
```

The first implementation will be `KdenliveProjectAdapter`. It should accept
the normalized media metadata and generated subtitle path already produced by
the caption pipeline, then write a project and its required sibling assets.
The service layer should not construct XML nodes directly.

## Minimal export spike

Before building the complete adapter, implement and manually verify a small
fixture-driven spike:

1. Generate a Gen 5/document 1.1 `.kdenlive` file for a short local MP4 and a
   generated SRT.
2. Parse the result with an XML parser and assert the profile, source
   producer, main-bin entry, timeline tractor, and subtitle filter are present.
3. Run local `melt` against the project and confirm that the source media can
   be rendered without missing-resource errors.
4. Open the project in the installed Kdenlive application and confirm the
   video, resolution/frame rate, synchronized captions, and subtitle text.
5. Close/save the project, inspect any Kdenlive rewrite, and record whether
   the generated structure was upgraded or normalized.
6. Add a Windows-path test covering spaces and backslashes in both the source
   video and project directory.

The spike should use a tiny test asset or a generated fixture and must not
download a model or require network access. A successful `melt` render is
useful evidence, but Kdenlive opening and displaying the editable subtitle
content is the acceptance test for this milestone.

## Risks and safeguards

- **Format drift:** record the target Kdenlive version and format generation;
  isolate XML writing behind the adapter and keep a representative fixture.
- **Path portability:** copy or place the SRT beside the project, use
  `pathlib`, and test paths containing spaces. Avoid shell command strings.
- **Media references:** never alter or overwrite the source video; fail with a
  typed project-export error when a referenced asset cannot be resolved.
- **Subtitle editability:** validate the SRT-backed subtitle track in the
  actual Kdenlive UI instead of treating a successful XML parse as sufficient.
- **External code provenance:** use outside projects as references until exact
  source reuse has passed license and maintenance review.

## Scope for the next implementation PR

The next PR should add the adapter boundary, a minimal Kdenlive project
writer, and tests for the generated XML. It should not add MCP handlers,
WebSocket control, GUI automation, or a full project editor. Those can be
considered only after the direct exporter passes the local Kdenlive acceptance
test.
