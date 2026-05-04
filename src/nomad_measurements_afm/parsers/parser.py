from nomad.datamodel.datamodel import EntryArchive
from nomad.parsing.parser import MatchingParser

from nomad_measurements_afm.schema_packages.schema_package import (
    ELNAFMMicroscopy
)

class NTMDTAFMParser(MatchingParser):
    def is_mainfile(
        self,
        filename: str,
        mime: str,
        buffer: bytes,
        decoded_buffer: str,
        compression: str = None,
    ) -> bool:
        """Gatekeeper for NT-MDT .mdt binary files."""
        if not super().is_mainfile(filename, mime, buffer, decoded_buffer, compression):
            return False

        # Verify the file extension
        if not filename.lower().endswith('.mdt'):
            return False

        # Verify the exact binary signature (first 4 bytes of an NT-MDT file are 01 B0 93 FF)
        if buffer and buffer.startswith(b"\x01\xb0\x93\xff"):
            return True

        return False

    def parse(
        self,
        mainfile: str,
        archive: EntryArchive,
        logger=None,
        child_archives=None,
    ) -> None:
        logger = logger or archive.m_context.logger

        # Instantiate the AFM schema
        entry = ELNAFMMicroscopy()
        entry.data_file = mainfile.rsplit('/', maxsplit=1)[-1]

        archive.data = entry

        entry.normalize(archive, logger)