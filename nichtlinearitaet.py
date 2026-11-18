"""2. Die strahlweise Nichtlinearit√§t Phi und ihre Jacobimatrix DPhi.

Arbeitsplan ¬ß¬ß1.1, 6.2:
    Phi(z) = -log(S exp(-M z)).
    DPhi(z) = diag(S exp(-M z))^(-1) S diag(exp(-M z)) M.

Achtung beim Lesen des Papers: Dort steht in Gleichung (20) +log.
Unser Phi ist also MINUS Phi aus dem Paper; entsprechend ist auch
DPhi das Negative der dortigen Ableitung. f entspricht X, R entspricht A.

Ein einzelner Strahl z hat shape (n_mat,).
Viele Strahlen Z haben shape (n_lin, n_mat).
Die Funktionen erlauben allgemein (..., n_mat); ... bleibt erhalten.
"""

import torch


def check_inputs(Z, S, M):
    """Fehlerhafte Dimensionen und nicht normierte Bins fr√ºh erkennen."""
    if M.ndim != 2 or S.ndim != 2 or Z.ndim < 1:
        raise ValueError('M und S m√ºssen Matrizen sein; Z braucht eine Materialachse.')
    if S.shape[1] != M.shape[0] or Z.shape[-1] != M.shape[1]:
        raise ValueError('Die Energie- oder Materialdimensionen passen nicht zusammen.')
    if not (Z.dtype == S.dtype == M.dtype and Z.device == S.device == M.device):
        raise ValueError('Z, S, M m√ºssen denselben dtype und dasselbe device haben.')
    if not Z.is_floating_point():
        raise ValueError('Bitte Gleitkommazahlen verwenden, vorzugsweise float64.')
    if not all(bool(torch.isfinite(A).all()) for A in (Z, S, M)):
        raise ValueError('Z, S, M m√ºssen endlich sein.')
    if bool((S < 0).any()) or bool((M < 0).any()):
        raise ValueError('S und M d√ºrfen keine negativen Eintr√§ge haben.')
    if not torch.allclose(S.sum(dim=1), torch.ones_like(S[:, 0])):
        raise ValueError('Jede Zeile von S muss auf Summe 1 normiert sein.')


def log_terms(Z, S, M):
    """Hilfsgröße a[...,b,e] = log(S[b,e]) - sum_m Z[...,m] M[e,m]."""
    check_inputs(Z, S, M)
    # S = 0 ist erlaubt: log(0) = -inf entfernt diese Energie aus dem Bin.
    # M wirkt auf die Materialachse: (..., n_mat) -> (..., n_spec).
    a = torch.log(S) - (Z @ M.T).unsqueeze(-2)
    # unsqueeze fügt die Bin-Achse ein. Broadcasting wiederholt die
    # Schwächung für jedes Bin, ohne eine Python-Schleife über Strahlen.
    return a


def phi(Z, S, M):
    """Phi(Z), shape (..., n_bin), stabil auch bei starker Schwächung.

    logsumexp(a) = c + log(sum(exp(a-c))), c = max(a).
    Damit vermeiden wir, zuerst winzige Intensitäten auszurechnen und
    anschließend log(0) zu erhalten. Es wird nichts künstlich geclippt.
    """
    a = log_terms(Z, S, M)
    return -torch.logsumexp(a, dim=-1)


def dphi(Z, S, M):
    """DPhi(Z), shape (..., n_bin, n_mat), explizit statt per Autodiff.

    w[...,b,e] = S[b,e] exp(-(M z)[e]) / sum_j S[b,j] exp(-(M z)[j]).
    DPhi[...,b,m] = sum_e w[...,b,e] M[e,m].
    Jede Ableitungszeile ist ein gewichteter Mittelwert der Spektren.
    """
    a = log_terms(Z, S, M)
    w = torch.softmax(a, dim=-1) #torch.softmax(a): exp(a_e) / [ sum_k exp(a_k) ]
    return w @ M




if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from spektralmodell import spectral_model

    # ============================================================
    # 1. Spektralmodell und Materialsinogramme vorbereiten
    # ============================================================

    # Zwei Materialien (Wasser, Knochen), drei Energiebins
    E, S, M = spectral_model(n_bin=3)

    # Wir betrachten 201 verschiedene Materialsinogramme.
    # Wasser wird variiert, Knochen bleibt konstant.

    Z = torch.zeros((201, 2), dtype=torch.float64)

    Z[:, 0] = torch.linspace(0, 20, 201)  # Wasser
    Z[:, 1] = 0.0                         # Knochen

    # Nichtlinearität und Jacobimatrix berechnen
    Y = phi(Z, S, M)
    J = dphi(Z, S, M)

    # Für die Darstellung in Matplotlib
    z_water = Z[:, 0].numpy()


    # ============================================================
    # 2. Nichtlinearität Phi und Linearisierung bei Null
    # ============================================================

    # Jacobimatrix bei z = 0
    z0 = torch.zeros(2, dtype=torch.float64)
    J0 = dphi(z0, S, M)

    # Linearisierung: Phi(z) ungefähr Dphi(0) @ z
    Y_linear = Z @ J0.T

    fig, ax = plt.subplots(figsize=(9, 5))

    for b in range(S.shape[0]):

        line, = ax.plot(
            z_water, Y[:, b].numpy(),
            label=f'Phi - Bin {b + 1}'
        )

        ax.plot(
            z_water, Y_linear[:, b].numpy(),
            '--', color=line.get_color(),
            label=f'Linearisiert - Bin {b + 1}'
        )

    ax.set(
        xlabel='Materiallinienintegral Wasser [g/cm²]',
        ylabel='Logarithmierter Messwert',
        title='Nichtlinearität Phi und Linearisierung'
    )

    ax.grid(True, alpha=0.3)
    ax.legend()


    # ============================================================
    # 3. Jacobimatrix Dphi in Abhängigkeit von Wasser
    # ============================================================

    # Zwei Teilgrafiken für die Ableitungen nach beiden Materialien
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.5),
        constrained_layout=True
    )

    material_names = ['Wasser', 'Knochen']

    for m in range(2):

        for b in range(S.shape[0]):

            axes[m].plot(
                z_water, J[:, b, m].numpy(),
                label=f'Bin {b + 1}'
            )

        axes[m].set(
            xlabel='Materiallinienintegral Wasser [g/cm²]',
            ylabel='Partielle Ableitung',
            title=f'Ableitung nach {material_names[m]}'
        )

        axes[m].grid(True, alpha=0.3)
        axes[m].legend()


    # ============================================================
    # 4. Effektive spektrale Gewichte bei zwei Strahlen
    # ============================================================

    # Strahl 1: keine Schwächung
    z_empty = torch.tensor([0., 0.], dtype=torch.float64)

    # Strahl 2: Schwächung durch Wasser
    z_attenuated = torch.tensor([15., 0.], dtype=torch.float64)

    # Gewichte mit Softmax berechnen
    w_empty = torch.softmax(
        log_terms(z_empty, S, M), dim=-1
    )

    w_attenuated = torch.softmax(
        log_terms(z_attenuated, S, M), dim=-1
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for b in range(S.shape[0]):

        line, = ax.plot(
            E.numpy(), w_empty[b, :].numpy(),
            '--', label=f'Bin {b + 1}: ohne Schwächung'
        )

        ax.plot(
            E.numpy(), w_attenuated[b, :].numpy(),
            color=line.get_color(),
            label=f'Bin {b + 1}: mit Schwächung'
        )

    ax.set(
        xlabel='Photonenenergie E [keV]',
        ylabel='Normiertes Spektralgewicht',
        title='Veränderung der Energiegewichte'
    )

    ax.grid(True, alpha=0.3)
    ax.legend()


    # ============================================================
    # 5. Kurze mathematische Kontrollen
    # ============================================================

    print('\n--- Nichtlinearität P1 ---')
    print(f'Dimension von Z:    {tuple(Z.shape)}')
    print(f'Dimension von Y:    {tuple(Y.shape)}')
    print(f'Dimension von J:    {tuple(J.shape)}')

    # Grenzfälle aus dem Arbeitsplan
    print(f'Phi(0) = 0:       {torch.allclose(phi(z0, S, M), torch.zeros(3, dtype=torch.float64))}')
    print(f'DPhi(0) = S @ M:  {torch.allclose(J0, S @ M)}')

    plt.show()
