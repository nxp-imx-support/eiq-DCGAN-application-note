# DCGAN inference
## Dependencies:
1. i.MX8MMini board running an image with the eIQ release.
2. Connect display to the board: HDMI display connected to the board with an IMX-MIPI-HDMI adapter.

## Compile on the host machine:
1. Extract toolchain with eIQ on the host machine.
2. Source the cross compiler environment. For example, 'source /opt/fsl-imx-internal-xwayland/5.4-zeus/environment-setup-aarch64-poky-linux'.
4. Copy the tflite model in the directory src/.
5. In src/ directory, run 'make -f Makefile.linux' to build the app.
6. Copy the built app "DCGAN" and the tflite model to the board and run it with command “./DCGAN”

## Link
For further information on this demo please check the Application Note document
