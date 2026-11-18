"""What ``stack.sh fetch`` reads from a fetched upstream tag (ADR 0212).

Dependency pins of a superbuild are read from the tag, never repeated in a
recipe: Palace's ``cmake/ExternalGitTags.cmake`` and ``download_patch`` calls,
a ``.gitmodules`` file, ``GIT_REPOSITORY``/``GIT_TAG`` pairs of a CMake file,
scivision/mumps's ``cmake/source.json``. Each command reads the file's text
from standard input (``git show TAG:FILE | python3 sources.py ...``) and
prints one record per line, fields separated by spaces. Runs on the cluster's
system Python: Python 3.9 and the standard library.

Command line (``python3 sources.py COMMAND ...``)::

    palace-git                 NAME URL TAG of each git dependency
    palace-tarballs            NAME FILE URL of each URL (tarball) dependency
    patches                    DIR FILE URL SHA256 of each download_patch call
    git-repositories           URL TAG of each GIT_REPOSITORY/GIT_TAG pair
    archive-git ARCHIVE        URL TAG of each pair in the CMake files of a tarball
    gitmodules                 PATH URL of each submodule
    mumps-tarball TAG          FILE URL SHA256 of the MUMPS release of a scivision tag
    mumps-release              FILE URL of the MUMPS release of a scivision CMakeLists
    url-tarballs               FILE URL MD5 of each URL/URL_MD5 download
    tarball-name URL           the mirror file name of a tarball URL
"""

from __future__ import annotations

import json
import re
import sys
import tarfile
import zipfile

_SET = re.compile(r"set\(\s*(\w+)\s+\"([^\"]*)\"", re.MULTILINE)


def _cache_strings(text: str) -> dict[str, str]:
    """The ``set(VAR "value" ...)`` assignments of a CMake file."""
    return dict(_SET.findall(text))


def palace_git(text: str) -> list[tuple[str, str, str]]:
    """The git dependencies of Palace's ``cmake/ExternalGitTags.cmake``.

    Returns:
        ``(NAME, URL, TAG)`` for each ``EXTERN_<NAME>_URL`` that has an
        ``EXTERN_<NAME>_GIT_TAG``, in file order (v0.18.1 spells Umpire's
        URL without ``.git``). The URL is spelled as the file spells it: the
        mirror path follows it.
    """
    values = _cache_strings(text)
    deps = []
    for var, url in values.items():
        match = re.fullmatch(r"EXTERN_(\w+)_URL", var)
        if match:
            name = match.group(1)
            tag = values.get(f"EXTERN_{name}_GIT_TAG")
            if tag:
                deps.append((name, url, tag))
    return deps


#: Archive extensions, longest first.
_EXTENSIONS = (".tar.gz", ".tar.xz", ".tar.bz2", ".tgz", ".zip")


def _split_extension(name: str) -> tuple[str, str]:
    for ext in _EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)], ext
    return name, ""


def tarball_name(url: str) -> str:
    """The file name of a tarball in the source mirror's ``tarballs/``.

    A name says project and version, so generic upstream names cannot clash:
    GitHub tag archives become ``<repo>-<tag><ext>``, Bitbucket ``get/``
    archives ``<owner>-<repo>-<tag><ext>``, release assets without a version
    in their name ``<stem>-<version><ext>``; any other URL keeps its basename.
    """
    path = url.split("://", 1)[-1].split("?", 1)[0]
    parts = path.split("/")
    basename = parts[-1]
    stem, ext = _split_extension(basename)
    if parts[0] == "github.com" and len(parts) >= 5:
        repo = parts[2]
        if parts[3] == "archive":
            return f"{repo}-{stem}{ext}"
        if parts[3:5] == ["releases", "download"] and len(parts) >= 7:
            version = parts[5].removeprefix("v")
            return basename if version in basename else f"{stem}-{version}{ext}"
    if parts[0] == "bitbucket.org" and len(parts) >= 5 and parts[3] == "get":
        return f"{parts[1]}-{parts[2]}-{stem}{ext}"
    return basename


def palace_tarballs(text: str) -> list[tuple[str, str, str]]:
    """The URL (tarball) dependencies of Palace's ``ExternalGitTags.cmake``.

    Returns:
        ``(NAME, FILE, URL)`` for each ``EXTERN_<NAME>_URL`` without an
        ``EXTERN_<NAME>_GIT_TAG`` (:func:`palace_git` takes those); ``FILE``
        is its name in the mirror (:func:`tarball_name`).
    """
    values = _cache_strings(text)
    tarballs = []
    for var, url in values.items():
        match = re.fullmatch(r"EXTERN_(\w+)_URL", var)
        if match and f"EXTERN_{match.group(1)}_GIT_TAG" not in values:
            tarballs.append((match.group(1), tarball_name(url), url))
    return tarballs


_DOWNLOAD_PATCH = re.compile(
    r"download_patch\(\s*\w+\s+\"([^\"]+)\"\s+(\S+)\s+\"([^\"]+)\"\s+([0-9a-f]{64})\s*\)"
)
#: Palace v0.18.1's MFEM-only form: ``download_mfem_patch(FILE "URL" SHA256)``
#: into ``${MFEM_PATCH_DIR}`` (a ``file(DOWNLOAD)`` with ``EXPECTED_HASH``, which
#: also skips a file already there with that hash).
_DOWNLOAD_MFEM_PATCH = re.compile(
    r"download_mfem_patch\(\s*(\S+)\s+\"([^\"]+)\"\s+([0-9a-f]{64})\s*\)"
)


def patches(text: str) -> list[tuple[str, str, str, str]]:
    """The ``download_patch`` calls of Palace's CMake files (``PatchHelpers.cmake``).

    Also v0.18.1's ``download_mfem_patch`` calls (:data:`_DOWNLOAD_MFEM_PATCH`).

    Palace downloads these at configure time into its build tree; a file that
    is already there with the expected hash is not downloaded again, so the
    build pre-places them from the mirror.

    Returns:
        ``(DIR, FILE, URL, SHA256)`` per call; ``DIR`` is relative to the
        build tree (``${CMAKE_BINARY_DIR}`` resolved through the file's
        ``set()`` calls).
    """
    values = _cache_strings(text)
    found = []
    prefix = "${CMAKE_BINARY_DIR}/"
    calls = _DOWNLOAD_PATCH.findall(text) + [
        ("${MFEM_PATCH_DIR}", name, url, digest)
        for name, url, digest in _DOWNLOAD_MFEM_PATCH.findall(text)
    ]
    for written, name, url, digest in calls:
        directory = written
        for _ in range(5):
            resolved = re.sub(
                r"\$\{(\w+)\}",
                lambda m: values.get(m.group(1), m.group(0)),
                directory,
            )
            if resolved == directory:
                break
            directory = resolved
        if not directory.startswith(prefix):
            msg = f"download_patch of {name}: {written} is not in the build tree"
            raise ValueError(msg)
        found.append((directory[len(prefix) :], name, url, digest))
    return found


_GIT_REPOSITORY = re.compile(r"GIT_REPOSITORY\s+(\S+)\s+GIT_TAG\s+(\S+)")


def git_repositories(text: str) -> list[tuple[str, str]]:
    """``(URL, TAG)`` of each ``GIT_REPOSITORY``/``GIT_TAG`` pair of a CMake file."""
    return [
        (url.strip('"'), tag.strip('"')) for url, tag in _GIT_REPOSITORY.findall(text)
    ]


def _is_cmake(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base == "CMakeLists.txt" or base.endswith(".cmake")


def archive_git_repositories(path: str) -> list[tuple[str, str]]:
    """The git dependencies a tarball's own CMake files clone when it configures.

    A URL dependency of a superbuild can fetch further sources itself
    (scnlib clones fast_float at configure time); the fetch mirrors them so
    the job-local ``insteadOf`` finds them.

    Returns:
        ``(URL, TAG)`` of each ``GIT_REPOSITORY``/``GIT_TAG`` pair in the
        archive's ``CMakeLists.txt`` and ``*.cmake`` files whose tag is
        literal (a ``${...}`` tag names a package found instead), in archive
        order, without repeats.
    """
    texts = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zipped:
            texts = [
                zipped.read(name).decode(errors="replace")
                for name in zipped.namelist()
                if _is_cmake(name)
            ]
    else:
        with tarfile.open(path) as tar:
            for member in tar:
                handle = tar.extractfile(member) if _is_cmake(member.name) else None
                if handle is not None:
                    texts.append(handle.read().decode(errors="replace"))
    found = []
    for text in texts:
        for pair in git_repositories(text):
            if "${" not in pair[1] and pair not in found:
                found.append(pair)
    return found


def gitmodules(text: str) -> list[tuple[str, str]]:
    """``(PATH, URL)`` of each submodule of a ``.gitmodules`` file, as spelled."""
    found = []
    path = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[submodule"):
            path = None
        elif "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            if key == "path":
                path = value
            elif key == "url" and path is not None:
                found.append((path, value))
    return found


def mumps_tarball(text: str, tag: str) -> list[tuple[str, str, str]]:
    """The MUMPS release a scivision/mumps tag builds, from its ``cmake/source.json``.

    scivision's ``CMakeLists.txt`` takes the release from the first three
    numbers of the project version (tag ``v5.9.1.2`` builds MUMPS 5.9.1) and
    its sha256 from ``mumps_sha256`` in ``source.json``.

    Returns:
        One ``(FILE, URL, SHA256)``.
    """
    match = re.match(r"v?(\d+\.\d+\.\d+)", tag)
    if match is None:
        msg = f"scivision/mumps tag {tag!r} names no MUMPS release"
        raise ValueError(msg)
    version = match.group(1)
    digests = json.loads(text)["mumps_sha256"]
    if version not in digests:
        msg = f"scivision/mumps {tag}: no sha256 of MUMPS {version} in source.json"
        raise ValueError(msg)
    name = f"MUMPS_{version}.tar.gz"
    return [(name, f"https://mumps-solver.org/{name}", digests[version])]


_UPSTREAM_VERSION = re.compile(r"set\(\s*MUMPS_UPSTREAM_VERSION\s+(\d+\.\d+\.\d+)\s*\)")


def mumps_release(text: str) -> list[tuple[str, str]]:
    """The MUMPS release a scivision/mumps ``CMakeLists.txt`` downloads.

    For a scivision commit without ``cmake/source.json`` (Palace v0.18.1 pins
    one): the default ``MUMPS_UPSTREAM_VERSION``. It carries no sha256; the
    recipe pins it.

    Returns:
        One ``(FILE, URL)``.
    """
    match = _UPSTREAM_VERSION.search(text)
    if match is None:
        msg = "scivision/mumps CMakeLists.txt sets no MUMPS_UPSTREAM_VERSION"
        raise ValueError(msg)
    name = f"MUMPS_{match.group(1)}.tar.gz"
    return [(name, f"https://mumps-solver.org/{name}")]


_URL_MD5 = re.compile(r"\bURL\s+(\S+)\s+URL_MD5\s+([0-9a-f]{32})")


def url_tarballs(text: str) -> list[tuple[str, str, str]]:
    """The ``URL``/``URL_MD5`` downloads of a CMake file (netgen's METIS).

    Returns:
        One ``(FILE, URL, MD5)`` per download, FILE as named in the mirror.
    """
    return [(tarball_name(url), url, md5) for url, md5 in _URL_MD5.findall(text)]


#: The command line: name, number of arguments after the name, handler. A
#: handler reads the file text from standard input unless it takes no text.
COMMANDS = {
    "palace-git": (0, palace_git),
    "palace-tarballs": (0, palace_tarballs),
    "patches": (0, patches),
    "git-repositories": (0, git_repositories),
    "gitmodules": (0, gitmodules),
    "mumps-tarball": (1, mumps_tarball),
    "mumps-release": (0, mumps_release),
    "url-tarballs": (0, url_tarballs),
}


def main(argv: list[str]) -> int:
    """Run one command of the command line."""
    if len(argv) == 2 and argv[0] == "tarball-name":
        print(tarball_name(argv[1]))
        return 0
    if len(argv) == 2 and argv[0] == "archive-git":
        for record in archive_git_repositories(argv[1]):
            print(" ".join(record))
        return 0
    if not argv or argv[0] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return 2
    count, handler = COMMANDS[argv[0]]
    if len(argv) - 1 != count:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        records = handler(sys.stdin.read(), *argv[1:])
    except ValueError as error:
        print(f"sources.py: {error}", file=sys.stderr)
        return 1
    for record in records:
        print(" ".join(record))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
