#!/bin/bash

# Stop script for LLM Trading Platform

echo "🛑 Stopping LLM Trading Platform..."

docker-compose down

echo "✅ All services stopped"
echo ""
echo "💡 To remove all data (database, cache, etc.):"
echo "   docker-compose down -v"
