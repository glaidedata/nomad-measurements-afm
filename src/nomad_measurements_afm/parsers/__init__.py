from nomad.config.models.plugins import ParserEntryPoint


class AFMParserEntryPoint(ParserEntryPoint):
    def load(self):
        from nomad_measurements_afm.parsers.parser import NTMDTAFMParser

        return NTMDTAFMParser(**self.dict())


parser_entry_point = AFMParserEntryPoint(
    name='NTMDTAFMParser',
    description='Parser for NT-MDT .mdt binary AFM files.',
    mainfile_name_re=r'^.*\.mdt$',
)