function [ber, numBits] = simula_qam3_gpu(EbNo, maxNumErrs, maxNumBits)
% Simulació Monte Carlo de 16-QAM sobre canal AWGN per a BERTool.
% Versió GPU (CUDA) - requereix Parallel Computing Toolbox + gràfica NVIDIA
% (recomanat, sèrie 20 cap amunt)
%
% Diferències respecte simula_qam3:
%   - Totes les operacions vectorials s'executen a la GPU via gpuArray
%   - Bloc de bits per iteració augmentat a 1.000.000 (vs 10.000 CPU)
%   - Límits augmentats per aprofitar la potència de calcul de la GPU
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
constel_symb = gpuArray([-3+3j, -1+3j, 1+3j, 3+3j, ...
                         -3+1j, -1+1j, 1+1j, 3+1j, ...
                         -3-1j, -1-1j, 1-1j, 3-1j, ...
                         -3-3j, -1-3j, 1-3j, 3-3j]); % 16-QAM

M = length(constel_symb); % 16
k = log2(M);              % 4 bits per simbol

% Matriu de bits numèrica (0/1) per compatibilitat GPU - Gray code 4 bits
constel_bits_num = gpuArray([0 0 0 0; 0 1 0 0; 1 1 0 0; 1 0 0 0; ...
                             0 0 0 1; 0 1 0 1; 1 1 0 1; 1 0 0 1; ...
                             0 0 1 1; 0 1 1 1; 1 1 1 1; 1 0 1 1; ...
                             0 0 1 0; 0 1 1 0; 1 1 1 0; 1 0 1 0]);

nBitsBloc = 50000000; % 50M bits per bloc
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

    % Modular
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
