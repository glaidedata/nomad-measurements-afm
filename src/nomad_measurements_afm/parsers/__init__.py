from nomad.config.models.plugins import ParserEntryPoint


class NTMDTAFMParserEntryPoint(ParserEntryPoint):
    def load(self):
        from .parser import NTMDTAFMParser

        return NTMDTAFMParser(**self.dict())


parser_entry_point = NTMDTAFMParserEntryPoint(
    name='NT-MDT AFM Parser',
    description='Parser for NT-MDT .mdt binary AFM files.',
    mainfile_name_re=r'^.*\.mdt$',
)
