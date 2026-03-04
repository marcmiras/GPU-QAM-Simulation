# Simulació QAM/PSK amb GPU (CUDA)

## Requisits

- MATLAB R2020b o superior
- Parallel Computing Toolbox
- GPU NVIDIA compatible amb CUDA (sèrie 20 o superior recomanada, 12 GB de VRAM o més)

## Configuració prèvia

1. Verificar que MATLAB detecta la GPU:
   ```matlab
   gpuDevice
   ```

2. **Important:** Desactivar el parallel pool per alliberar VRAM.
   A BERTool: **Acceleration → Parallel pool → Off**

3. Si hi ha un pool actiu, eliminar-lo:
   ```matlab
   delete(gcp('nocreate'))
   ```

## Execució amb BERTool

1. Obrir BERTool:
   ```matlab
   bertool
   ```

2. A la pestanya **Monte Carlo**, configurar:
   - **Eb/No range**: el rang desitjat (ex: `0:1:12`)
   - **Number of errors**: `1000` (recomanat)
   - **Number of bits**: `1e9` (recomanat)
   - **Function name**: `simula_qam1_gpu`, `simula_qam2_gpu` o `simula_qam3_gpu`

3. Assegurar que la carpeta `gpu/` està al path de MATLAB:
   ```matlab
   addpath('gpu')
   ```

4. Clicar **Run**.

## Notes

- Els blocs de bits per iteració (`nBitsBloc`) estan dimensionats per a GPUs de 12 GB de VRAM. Si es disposa de 16 GB o més, es poden incrementar els valors de `nBitsBloc` als fitxers `simula_qam*_gpu.m` per obtenir millor rendiment.
