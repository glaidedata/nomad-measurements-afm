import re

from nomad.datamodel.datamodel import EntryArchive
from nomad.parsing.parser import MatchingParser

# Import both of our specialized schemas!
from nomad_measurements_afm.schema_packages.schema_package import (
    ELNBrukerMicroscopy,
    ELNNTMDTMicroscopy,
)


class AFMParser(MatchingParser):
    def is_mainfile(
        self,
        filename: str,
        mime: str,
        buffer: bytes,
        decoded_buffer: str,
        compression: str = None,
    ) -> bool:
        """Gatekeeper for both NT-MDT and Bruker AFM files."""

        filename_lower = filename.lower()

        # 1. NT-MDT Check (.mdt + Hex Signature)
        if filename_lower.endswith('.mdt'):
            if buffer and buffer.startswith(b'\x01\xb0\x93\xff'):
                return True

        # 2. Bruker Check (.spm or .001, .002, .003, etc. + ASCII Signature)
        # Using a regex to catch any 3-digit extension
        is_bruker_ext = filename_lower.endswith('.spm') or re.search(
            r'\.\d{3}$', filename_lower
        )
        if is_bruker_ext:
            # Bruker files universally start with this exact string
            if buffer and buffer.startswith(b'\\*File list'):
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

        # Extract just the filename from the path
        filename = mainfile.rsplit('/', maxsplit=1)[-1]
        filename_lower = filename.lower()

        # Route to the correct Schema based on the file extension
        if filename_lower.endswith('.mdt'):
            entry = ELNNTMDTMicroscopy()
        elif filename_lower.endswith('.spm') or re.search(r'\.\d{3}$', filename_lower):
            entry = ELNBrukerMicroscopy()
        else:
            logger.error(f'Unsupported AFM file format: {filename}')
            return

        # Assign the file and attach the schema to the archive
        entry.data_file = filename
        archive.data = entry

        # Trigger the reader inside the schema's normalize function
        entry.normalize(archive, logger)
