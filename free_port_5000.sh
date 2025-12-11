#!/bin/bash
# Script para liberar el puerto 5000 temporalmente
# Nota: AirPlay Receiver se reiniciará automáticamente después de unos segundos

echo "Buscando procesos usando el puerto 5000..."
PIDS=$(lsof -ti:5000)

if [ -z "$PIDS" ]; then
    echo "✓ El puerto 5000 está libre"
    exit 0
fi

echo "Procesos encontrados: $PIDS"
echo "Deteniendo procesos..."

for PID in $PIDS; do
    COMMAND=$(ps -p $PID -o comm= 2>/dev/null)
    if [[ "$COMMAND" == *"ControlCenter"* ]]; then
        echo "  - Deteniendo ControlCenter (AirPlay Receiver) - PID: $PID"
        kill $PID 2>/dev/null
    else
        echo "  - Deteniendo proceso $COMMAND - PID: $PID"
        kill $PID 2>/dev/null
    fi
done

sleep 1

# Verificar si el puerto está libre ahora
REMAINING=$(lsof -ti:5000)
if [ -z "$REMAINING" ]; then
    echo "✓ Puerto 5000 liberado exitosamente"
    echo "⚠️  Nota: AirPlay Receiver se reiniciará automáticamente en unos segundos"
    echo "   Para desactivarlo permanentemente, sigue las instrucciones en DESACTIVAR_AIRPLAY.md"
else
    echo "⚠️  Algunos procesos aún están usando el puerto 5000: $REMAINING"
    echo "   Puede que necesites desactivar AirPlay Receiver desde System Settings"
fi


