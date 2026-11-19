"""1. Spektralmodell: Energiestützstellen E und Matrizen S, M.

Notation aus dem Arbeitsplan, Abschnitt 1.1:
    M[e,m] = mu_m(E[e]),       shape: (n_spec, n_mat)
    S[b,e] = s_b(E[e]) * dE,   shape: (n_bin, n_spec)
    sum_e S[b,e] = 1.


Hier sind mu_m Massenschwächungskoeffizienten in cm^2/g.
Die Materialbilder f enthalten daher Dichten in g/cm^3.
R enthält Weglängen in cm, Z = R f hat die Einheit g/cm^2.
Damit ist Z @ M.T dimensionslos, wie für exp(-Z @ M.T) nötig.
"""

import csv
from pathlib import Path

import torch


def read_table(filename):
    """CSV-Zahlen lesen; alle numerischen Arrays sind torch-Tensoren.

    Path(__file__) sorgt dafür, dass die Daten auch gefunden werden,
    wenn Spyder gerade ein anderes Arbeitsverzeichnis verwendet.
    """
    path = Path(__file__).resolve().parent / 'raw' / filename
    with path.open(encoding='utf-8') as stream:
        rows = list(csv.reader(stream))[1:]  # Erste Zeile: Spaltennamen.
    return torch.tensor([[float(v) for v in row] for row in rows],
                        dtype=torch.float64)


def attenuation(E):
    """NIST-Werte für Wasser und kortikalen Knochen, log-log interpoliert.
    
    Laut Arbeitsplan §1.1 (S.3) gilt M_{e,m} = mu_m(E_e)
    E_e die Energie an der e-ten Energiestützstelle.
    mu_m(E) der Massenschwächungskoeffizient des Materials m bei Energie E.
    
    E: PyTorch-Tensor, enthält die Energien, für diese berechne Schwächungskoeffizienten."""
    
    
    table = read_table('nist_water_bone.csv').to(device=E.device, dtype=E.dtype)
    """ to(device=E.device, dtype=E.dtype):  
            Eingelesene Daten verwenden selben Datentyp und selbes Rechengerät wie E
            Später wichtig, wenn statt CPU mit GPU gearbeitet wird"""
        
    
    E_table, M_table = table[:, 0].contiguous(), table[:, 1:]
    """ E_table sind die Energiewerte der NIST-Tabelle; E Energien, mit denen wir arbeiten möchten
    M_table wählt dann alle Spalten ab der 2. Spalte. Zuerst 2 Materialien => 2 Spalten
    Warum reicht M_table nicht? Die .csv-Datei enthält ggf. nicht die gewünschten Energien
    Müssen also interpolieren aus bekannten Daten
    .contiguous():      Stellt sicher, dass Werte zusammenhängend gespeichert werden"""
    
    
    if not bool(torch.all(torch.isfinite(E) & (E >= 20) & (E <= 150))):
        raise ValueError('E muss endlich sein und zwischen 20 und 150 keV liegen.')
    """ Überprüft lediglich, ob unsere gewünschten Energiewerte den voraussetzungen entsprechen"""
    

    # right ist der rechte Nachbar, left der linke Nachbar in der Tabelle.
    right = torch.searchsorted(E_table, E.contiguous()).clamp(1, len(E_table) - 1)
    """ E_table = torch.tensor([20., 30., 40., 50., 60.])
        E = torch.tensor([25., 42.])
        right = torch.searchsorted(E_table, E)
        print(right)
        Ausgabe: tensor([1, 3])
        
        .clamp begrenzt die Werte eines Tensors auf ein vorgegebenes Intervall"""
    
        
    left = right - 1
    
    
    #Logarithmische Interpolation
    t = ((torch.log(E) - torch.log(E_table[left]))
         / (torch.log(E_table[right]) - torch.log(E_table[left])))
    
    log_M = ((1 - t[:, None]) * torch.log(M_table[left])
             + t[:, None] * torch.log(M_table[right]))
    """ Formel: mu_m(E_e) = exp((1-t_e) ln(mu_m(E_left,e)) + t_e ln(mu_m(E_right,e)))
        t[:, None]: fügt eine weitere Dimension hinzu
        Angenommen t = torch.tensor([0.3, 0.5, 0.7]), dann:
            torch.Size([3]) bevor,
            torch.Size([3, 1]) danach.
            *: Elementweise Multiplikation"""
    
    return torch.exp(log_M) #exp agiert elementweise


def spectral_model(n_bin=3):
    """Das P1-Modell für zwei Materialien und zwei oder drei ideale Bins.

    E = 20, 21, ..., 120 keV; dE = 1 keV.
    2 Bins: [20,60), [60,121).
    3 Bins: [20,50), [50,80), [80,121).
    121 ist die exklusive obere Grenze; letzte St√ºtzstelle ist 120 keV.

    I0 ist das tabellierte Referenzspektrum, kein erfundenes Röhrenmodell.
    Seine absolute Skalierung fällt durch die Normierung pro Bin weg.
    Quelle, Umwandlung und Einheiten: data/QUELLEN.md.
    """
    if n_bin not in (2, 3):
        raise ValueError('P1 verwendet zwei oder drei Bins.')
        
    table = read_table('source_spectrum.csv')
    
    selected = (table[:, 0] >= 20) & (table[:, 0] <= 120)
    E, I0 = table[selected, 0], table[selected, 1]
    
    M = attenuation(E)
    
    edges = [20, 60, 121] if n_bin == 2 else [20, 50, 80, 121]

    S = torch.zeros((n_bin, len(E)), dtype=torch.float64)
    for b in range(n_bin):
        inside = (E >= edges[b]) & (E < edges[b + 1])
        # Idealer Detektor: Sensitivität 1 innerhalb, 0 außerhalb des Bins.
        weights = I0 * inside * 1.0  # 1.0 ist delta E in keV.
        S[b, :] = weights / weights.sum() 
        """"Normierung für phi, falls """
    return E, S, M



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # ============================================================
    # 1. Referenzspektrum I0(E) aus der CSV-Datei
    # ============================================================

    table = read_table('source_spectrum.csv')

    E_source = table[:, 0]       # Photonenenergien in keV
    I0 = table[:, 1]             # Tabelliertes Referenzspektrum

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(E_source.numpy(), I0.numpy(), color='steelblue')

    ax.set(
        xlabel='Photonenenergie E [keV]',
        ylabel='Referenzspektrum I0(E)',
        title='Referenzspektrum der Röntgenstrahlung'
    )

    ax.grid(True, alpha=0.3)


    # ============================================================
    # 2. Massenschwächungskoeffizienten von Wasser und Knochen
    # ============================================================

    # Ursprüngliche NIST-Werte
    table = read_table('nist_water_bone.csv')

    E_table = table[:, 0]
    M_table = table[:, 1:]

    # Feines Energiegitter für die interpolierten Kurven
    E_fine = torch.linspace(20, 150, 500, dtype=torch.float64)

    # Log-Log-Interpolation an den 500 Energiestützstellen
    M_fine = attenuation(E_fine)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Wasser
    ax.plot(E_fine.numpy(), M_fine[:, 0].numpy(),
            color='steelblue', label='Wasser (interpoliert)')

    ax.plot(E_table.numpy(), M_table[:, 0].numpy(),
            'o', color='steelblue', markersize=4,
            label='Wasser (NIST)')

    # Knochen
    ax.plot(E_fine.numpy(), M_fine[:, 1].numpy(),
            color='darkorange', label='Knochen (interpoliert)')

    ax.plot(E_table.numpy(), M_table[:, 1].numpy(),
            'o', color='darkorange', markersize=4,
            label='Knochen (NIST)')

    ax.set(
        xlabel='Photonenenergie E [keV]',
        ylabel='Massenschwächungskoeffizient [cm²/g]',
        title='Massenschwächungskoeffizienten'
    )

    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()


    # ============================================================
    # 3. Vollständiges Spektralmodell: Matrizen S und M
    # ============================================================

    n_bin = 3

    E, S, M = spectral_model(n_bin=n_bin)

    # Alle normierten Bin-Spektren darstellen
    fig, ax = plt.subplots(figsize=(9, 5))

    for b in range(n_bin):
        ax.plot(E.numpy(), S[b, :].numpy(),
                label=f'Bin {b + 1}')

    ax.set(
        xlabel='Photonenenergie E [keV]',
        ylabel='Normiertes Spektralgewicht',
        title='Effektive Spektren der Energiebins'
    )

    ax.grid(True, alpha=0.3)
    ax.legend()


    # ============================================================
    # 4. Kurze Kontrolle der Matrixdimensionen und Normierung
    # ============================================================

    print('\n--- Spektralmodell P1 ---')
    print(f'Anzahl Energien:     {len(E)}')
    print(f'Anzahl Materialien:  {M.shape[1]}')
    print(f'Anzahl Bins:         {S.shape[0]}')
    print(f'Dimension von S:     {tuple(S.shape)}')
    print(f'Dimension von M:     {tuple(M.shape)}')
    print(f'Zeilensummen von S:  {S.sum(dim=1).tolist()}')

    plt.show()
