import re

from nomad.datamodel.context import ServerContext
from nomad.datamodel.datamodel import EntryArchive
from nomad.parsing.parser import MatchingParser
from nomad_measurements.utils import create_archive

from nomad_measurements_afm.schema_packages.schema_package import (
    ELNBrukerMicroscopy,
    ELNNTMDTMicroscopy,
    RawFileAFMData,
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
        is_bruker_ext = filename_lower.endswith('.spm') or re.search(
            r'\.\d{3}$', filename_lower
        )
        if is_bruker_ext:
            # Accept both Image scans (\*File list) and Force curves (\*Force file list)
            if buffer and (
                buffer.startswith(b'\\*File list')
                or buffer.startswith(b'\\*Force file list')
            ):
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

        # Extract the filename, handling server context paths correctly
        data_file = mainfile.rsplit('/', maxsplit=1)[-1]
        if isinstance(archive.m_context, ServerContext):
            data_file = mainfile.split('/raw/', 1)[1]

        filename_lower = data_file.lower()

        # Route to the correct Schema based on the file extension
        if filename_lower.endswith('.mdt'):
            entry = ELNNTMDTMicroscopy.m_from_dict(ELNNTMDTMicroscopy.m_def.a_template)
        elif filename_lower.endswith('.spm') or re.search(r'\.\d{3}$', filename_lower):
            entry = ELNBrukerMicroscopy.m_from_dict(
                ELNBrukerMicroscopy.m_def.a_template
            )
        else:
            logger.error(f'Unsupported AFM file format: {data_file}')
            return

        # Assign the file name to the entry
        entry.data_file = data_file

        # Create the separate editable .archive.json file to preserve ELN edits
        archive_name = f'{"".join(data_file.split(".")[:-1])}.archive.json'

        # Link the raw file to the generated ELN using the placeholder
        archive.data = RawFileAFMData(
            measurement=create_archive(entry, archive, archive_name)
        )

        # Clean up the display name in the GUI
        archive.metadata.entry_name = f'{data_file} data file'
