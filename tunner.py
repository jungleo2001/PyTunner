import numpy as np
import sounddevice as sd
from scipy.fftpack import fft

# Frequências das cordas do violão (em Hz)
CORDAS_VIOLAO = {
    "E4": 329.63,  # Mi agudo (1ª corda)
    "B3": 246.94,  # Si (2ª corda)
    "G3": 196.00,  # Sol (3ª corda)
    "D3": 146.83,  # Ré (4ª corda)
    "A2": 110.00,  # Lá (5ª corda)
    "E2": 82.41    # Mi grave (6ª corda)
}

# Configurações
SAMPLE_RATE = 44100  # Taxa de amostragem (Hz)
DURATION = 1.0       # Duração da gravação (segundos)
WINDOW_SIZE = int(SAMPLE_RATE * DURATION)

def calcular_frequencia_dominante(audio_data, sample_rate):
    """
    Calcula a frequência dominante no áudio usando FFT.
    """
    # Aplica a FFT no áudio
    fft_result = fft(audio_data)
    freqs = np.fft.fftfreq(len(fft_result), 1 / sample_rate)

    # Obtém as magnitudes (valores absolutos)
    magnitudes = np.abs(fft_result)

    # Encontra a frequência dominante (ignora frequências negativas)
    freq_dominante = freqs[np.argmax(magnitudes[:len(magnitudes) // 2])]
    return abs(freq_dominante)  # Retorna o valor absoluto

def encontrar_corda_proxima(frequencia):
    """
    Encontra a corda mais próxima da frequência detectada.
    """
    # Calcula a diferença entre a frequência detectada e as frequências das cordas
    diferencas = {corda: abs(frequencia - freq) for corda, freq in CORDAS_VIOLAO.items()}
    corda_proxima = min(diferencas, key=diferencas.get)
    return corda_proxima, diferencas[corda_proxima]

def main():
    print("Afinador de Violão - Pressione Ctrl+C para parar.")
    print("Toque uma corda do violão para afinar...")

    try:
        while True:
            # Grava áudio do microfone
            print("\nGravando...")
            audio_data = sd.rec(int(WINDOW_SIZE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()  # Aguarda o fim da gravação

            # Remove o ruído do silêncio
            audio_data = audio_data[:, 0]  # Pega apenas o primeiro canal (mono)
            audio_data = audio_data - np.mean(audio_data)  # Remove o offset

            # Calcula a frequência dominante
            freq_dominante = calcular_frequencia_dominante(audio_data, SAMPLE_RATE)
            print(f"Frequência detectada: {freq_dominante:.2f} Hz")

            # Encontra a corda mais próxima
            corda, diferenca = encontrar_corda_proxima(freq_dominante)
            print(f"Corda mais próxima: {corda} ({CORDAS_VIOLAO[corda]} Hz)")

            # Verifica se a corda está afinada
            if diferenca < 1.0:  # Margem de erro de 1 Hz
                print("Corda afinada corretamente!")
            else:
                if freq_dominante > CORDAS_VIOLAO[corda]:
                    print("A corda está alta. Afrouxe um pouco.")
                else:
                    print("A corda está baixa. Aperte um pouco.")

    except KeyboardInterrupt:
        print("\nAfinador encerrado.")

if __name__ == "__main__":
    main()
