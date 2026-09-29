"""Domain errors raised by the use cases. The API maps them to HTTP statuses; the worker to `failed`."""


class VectorlessRagError(Exception):
    """Base class, so boundaries can tell domain errors from bugs."""


class DocumentNotFound(VectorlessRagError):
    """No such document for this user. Also used for other users' documents, so existence never leaks."""


class DuplicateDocument(VectorlessRagError):
    """A repository already holds this (user_id, file_sha256); the upload returns the existing record."""


class DocumentNotReady(VectorlessRagError):
    """The document exists but is not `completed` yet."""


class UnsupportedFile(VectorlessRagError):
    """Not a PDF."""


class FileTooLarge(VectorlessRagError):
    """Over MAX_UPLOAD_MB."""


class TooManyPages(VectorlessRagError):
    """Over MAX_PAGES."""


class ScannedDocument(VectorlessRagError):
    """Most pages have no text layer (spec 9.5); out of scope for v1."""


class ViewPagesRejected(VectorlessRagError):
    """view_pages cannot show what the agent asked for; the message goes back to the agent."""
