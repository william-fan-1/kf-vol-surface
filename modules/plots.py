from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_skews(
    df,
    date,
):
    fig, ax = plt.subplots(figsize=(9, 6))

    for dte in sorted(df['target_dte'].unique()):

        smile = (
            df[
                (df['date'] == date)
                & (df['target_dte'] == dte)
            ]
            .sort_values('target_moneyness')
        )

        ax.plot(
            smile['target_moneyness'],
            smile['IV'],
            marker='o',
            label=f'{dte} DTE',
        )

    ax.set_title(f'Volatility Smiles\n{date}')
    ax.set_xlabel('Moneyness')
    ax.set_ylabel('Implied Volatility')

    ax.legend()
    ax.grid(True)

    plt.show()

def plot_term_structures(
    df,
    date,
):
    fig, ax = plt.subplots(figsize=(9, 6))

    for m in sorted(df['target_moneyness'].unique()):

        term = (
            df[
                (df['date'] == date)
                & (df['target_moneyness'] == m)
            ]
            .sort_values('target_dte')
        )

        ax.plot(
            term['target_dte'],
            term['IV'],
            marker='o',
            label=f'{m:.2f}',
        )

    ax.set_title(f'Term Structures\n{date}')

    ax.set_xlabel('Days to Expiration')
    ax.set_ylabel('Implied Volatility')

    ax.legend(title='Moneyness')

    ax.grid(True)

    plt.show()

def plot_surface(surface):

    X, Y = np.meshgrid(
        surface.columns,
        surface.index
    )

    Z = surface.values

    fig = plt.figure(figsize=(10,7))

    ax = fig.add_subplot(
        111,
        projection='3d'
    )

    ax.plot_surface(
        X,
        Y,
        Z
    )

    ax.set_xlabel('Moneyness')
    ax.set_ylabel('DTE')
    ax.set_zlabel('IV')

    plt.show()