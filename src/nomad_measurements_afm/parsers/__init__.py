from nomad.config.models.plugins import ParserEntryPoint


class AFMParserEntryPoint(ParserEntryPoint):
    def load(self):
        from nomad_measurements_afm.parsers.parser import AFMParser

        return AFMParser(**self.dict())


parser_entry_point = AFMParserEntryPoint(
    name='AFM Parser',
    description='Parser for NT-MDT (.mdt) and Bruker (.003) AFM files.',
    mainfile_name_re=r'^.*\.(mdt|\d{3})$',
)
