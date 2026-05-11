from nomad.config.models.plugins import SchemaPackageEntryPoint


class AFMSchemaPackageEntryPoint(SchemaPackageEntryPoint):
    def load(self):
        from nomad_measurements_afm.schema_packages.schema_package import m_package

        return m_package


schema_package_entry_point = AFMSchemaPackageEntryPoint(
    name='AFM Schema',
    description='Schema package for Atomic Force Microscopy measurements (NT-MDT and Bruker).',
)
