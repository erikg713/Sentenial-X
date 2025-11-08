#!/bin/bash
mkdir -p sentenialx/telemetry/sinks sentenialx/telemetry/alerts
mv telemetry.py sentenialx/telemetry/collector.py
cat > sentenialx/telemetry/__init__.py << 'EOF'
from .collector import TelemetryCollector, emit_telemetry
from .schema import TelemetryRecord  # if you split schema out
__all__ = ["TelemetryCollector", "emit_telemetry", "TelemetryRecord"]
EOF

# Fix imports
find sentenialx -type f -name "*.py" -exec sed -i.bak 's/from telemetry import/from sentenialx.telemetry import/g' {} \; 2>/dev/null
find sentenialx -type f -name "*.bak" -delete

echo "✅ Telemetry now lives safely in sentenialx/telemetry/collector.py"
echo "   Import with: from sentenialx.telemetry import emit_telemetry"
