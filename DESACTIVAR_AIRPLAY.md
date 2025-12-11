# Cómo desactivar AirPlay Receiver en macOS para liberar el puerto 5000

AirPlay Receiver en macOS usa el puerto 5000 por defecto, lo que puede causar conflictos con aplicaciones Flask.

## Método 1: Desde System Settings (Recomendado)

1. Abre **System Settings** (Configuración del Sistema)
2. Ve a **General** (General)
3. Busca **AirDrop & Handoff** (AirDrop y Pase)
4. Desactiva **AirPlay Receiver** (Receptor de AirPlay)

## Método 2: Desde la línea de comandos

```bash
# Desactivar AirPlay Receiver
sudo defaults write com.apple.controlcenter.plist AirplayRecieverEnabled -bool false

# Reiniciar ControlCenter para aplicar los cambios
killall ControlCenter
```

## Método 3: Temporal (solo para esta sesión)

Si solo necesitas el puerto 5000 temporalmente, puedes matar el proceso:

```bash
# Encontrar el proceso
lsof -ti:5000

# Matar el proceso (se reiniciará automáticamente)
kill $(lsof -ti:5000)
```

**Nota:** Con el método 3, el proceso se reiniciará automáticamente después de unos segundos.

## Verificar que el puerto está libre

```bash
lsof -i:5000
```

Si no muestra ningún resultado, el puerto está libre.


