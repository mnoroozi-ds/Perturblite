"""Specification for the external PCAP-to-CSV preparation pipeline.

Model training starts from the final 1,481-feature CSV. PCAP parsing requires
packet/flow labeling context and remains separate from the learning code. See
``data/README.md`` for the complete preparation and schema requirements.
"""

MAX_RAW_PACKET_BYTES = 1594
PREPARED_PACKET_FEATURES = 1481

REMOVED_PACKET_FIELDS = (
    "ethernet_header",
    "ip_version",
    "ip_differentiated_services",
    "ip_protocol",
    "ip_source_address",
    "ip_destination_address",
    "tcp_source_port",
    "tcp_destination_port",
    "ip_options",
    "tcp_options",
    "ip_checksum",
    "tcp_checksum",
)
