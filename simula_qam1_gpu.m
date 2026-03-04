function [ber, numBits] = simula_qam1_gpu(EbNo, maxNumErrs, maxNumBits)
% Simulació Monte Carlo de 4-QAM sobre canal AWGN per a BERTool.
% Versió GPU (CUDA) - requereix Parallel Computing Toolbox + targeta
% gràfica de NVIDIA (recomanat sèrie 20 cap amunt)
%
% Diferències respecte simula_qam1:
%   - Totes les operacions vectorials s'executen a la GPU via gpuArray
%   - Bloc de bits per iteració augmentat a 1.000.000 (vs 10.000 CPU)
%   - Límits augmentats per aprofitar la potència de càlcul de la GPU
%   - demodqam_gpu vectoritzat
% 
% Marc Miras - Març 2026
%
% Requereix: Parallel Computing Toolbox + GPU CUDA compatible.

narginchk(3,3)

% Initialize variables related to exit criteria.
totErr  = 0; % Number of errors observed
numBits = 0; % Number of bits processed

% Utilitzem els limits de BERTool directament

% Constel·lació per bits
constel_symb = gpuArray([1+1i, 1-1i, -1-1i, -1+1i]); % 4-QAM

M = length(constel_symb); % 4
k = log2(M);              % 2 bits per símbol

% Matriu de bits numèrica (0/1) per compatibilitat GPU
constel_bits_num = gpuArray([0 0; 0 1; 1 1; 1 0]); % Gray code

nBitsBloc = 120000000; % 120M bits per bloc
nSymbolsBloc = nBitsBloc / k;

EbNo_lin = 10^(EbNo/10);
Ps = gather(mean(abs(constel_symb).^2));
Eb = Ps / k;
Pn = Eb / EbNo_lin;

% Assegurem que constel_symb és vector columna per demodqam_gpu
constel_symb_col = transpose(constel_symb);

while ((totErr < maxNumErrs) && (numBits < maxNumBits))

    if isBERToolSimulationStopped()
        break
    end

    % Generar símbols aleatoris directament a la GPU
    RandSymb = randi([1 M], 1, nSymbolsBloc, 'gpuArray');

    % Modulació
    mod = constel_symb(RandSymb);

    % Generar soroll AWGN a la GPU
    ruido = randn(1, nSymbolsBloc, 'gpuArray') + 1i*randn(1, nSymbolsBloc, 'gpuArray');
    ruidoFinal = sqrt(Pn/2) * ruido;

    simRecibido = mod + ruidoFinal;

    % Demodular a la GPU
    [~, nerrors] = demodqam_gpu(simRecibido, constel_symb_col, constel_bits_num, RandSymb);

    totErr  = totErr + nerrors;
    numBits = numBits + nBitsBloc;

end

% Compute the BER.
ber = totErr / numBits;
