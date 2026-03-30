#!/bin/bash
# monitor.sh
#
# Run from your laptop to check the health of the fraud API server.
# Prints: container status, resource usage, recent logs, health endpoint.
#
# Usage:
#   chmod +x monitor.sh
#   SERVER_IP=YOUR_SERVER_IP ./monitor.sh
#
# Or set the variable at the top:

SERVER_IP="${SERVER_IP:-YOUR_SERVER_IP}"

echo "============================================================"
echo "  FRAUD API — Server Monitoring Report"
echo "  Server: ${SERVER_IP}"
echo "============================================================"

ssh root@${SERVER_IP} << 'EOF'

  echo ""
  echo "── 1. CONTAINER STATUS ─────────────────────────────────────"
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" \
    --filter "name=fraud-api"

  echo ""
  echo "── 2. RESOURCE USAGE (CPU + Memory) ────────────────────────"
  docker stats fraud-api --no-stream \
    --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

  echo ""
  echo "── 3. LAST 20 LOG LINES ────────────────────────────────────"
  docker logs fraud-api --tail 20 2>&1

  echo ""
  echo "── 4. HEALTH CHECK ─────────────────────────────────────────"
  curl -s http://localhost:8000/ | python3 -m json.tool

EOF

echo ""
echo "── 5. EXTERNAL HEALTH CHECK (from your machine) ────────────"
curl -s http://${SERVER_IP}:8000/ | python3 -m json.tool

echo ""
echo "============================================================"
echo "  Docs: http://${SERVER_IP}:8000/docs"
echo "============================================================"
