import numpy as np
import plotly.graph_objects as go

def plotly_correlation_heatmap(corr_matrix, title="Matriz de Correlação"):
    """
    Plota a matriz de correlação usando plotly.
    corr_matrix: DataFrame do pandas (preferencial).
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            colorbar=dict(title='Correlação'),
            text=np.round(corr_matrix.values, 2),
            hovertemplate='(%{x}, %{y}): %{z:.2f}<extra></extra>'
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=45, side='top'),
        yaxis=dict(autorange='reversed'),  # Mantém diagonal principal de cima para baixo
        width=600,
        height=500
    )
    # Adiciona os valores no centro dos quadrados
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            fig.add_annotation(
                x=corr_matrix.columns[j],
                y=corr_matrix.index[i],
                text=f"{corr_matrix.values[i, j]:.2f}",
                showarrow=False,
                font=dict(color="black" if abs(corr_matrix.values[i, j]) < 0.5 else "white"),
                xanchor="center",
                yanchor="middle"
            )
    fig.show()

