from dataclasses import dataclass
from pathlib import PurePath

CODE_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".kt": "kotlin",
    ".scala": "scala",
    ".swift": "swift",
    ".sql": "sql",
    ".sh": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
}

DOC_TYPES: dict[str, str] = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".rst": "text",
}


@dataclass(frozen=True)
class FileKind:
    file_type: str  # pdf | markdown | text | code
    language: str | None
    extension: str


def detect_file_kind(filename: str) -> FileKind | None:
    ext = PurePath(filename).suffix.lower()
    if ext in DOC_TYPES:
        return FileKind(DOC_TYPES[ext], None, ext)
    if ext in CODE_LANGUAGES:
        return FileKind("code", CODE_LANGUAGES[ext], ext)
    return None


def supported_extensions() -> list[str]:
    return sorted({*DOC_TYPES, *CODE_LANGUAGES})
