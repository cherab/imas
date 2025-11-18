"""Serve as the IMAS sample dataset registry."""

registry = {
    "iter_disruption_113112_1.nc": "md5:f8747539b8d5a4b6974cc1bb33f2924a",  # JOREK 3D simulation
    "iter_scenario_123364_1.nc": "md5:b969c91e2c0f2df9a500edd55829764b",  # SOLPS-ITER simulation
    "iter_scenario_53298_seq1_DD4.nc": "md5:6bd52d3dd9b456e86aab6de5c0e0788a",  # JINTRAC simulation
}

# dataset method mapping with their associated filenames
# <method_name> : ["filename1", "filename2", ...]
method_files_map = {
    "iter_jintrac": ["iter_scenario_53298_seq1_DD4.nc"],
    "iter_solps": ["iter_scenario_123364_1.nc"],
    "iter_jorek": ["iter_disruption_113112_1.nc"],
}
