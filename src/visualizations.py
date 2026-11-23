import plotly.graph_objects as go

def plot_family_radar(family_scores: dict) -> go.Figure:
    """
    Génère un diagramme radar des scores d'affinité sémantique
    pour les 8 familles olfactives (entre 0 et 1).
    """
    families = list(family_scores.keys())
    scores = list(family_scores.values())

    # Fermer la boucle du polygone radar
    families_closed = families + [families[0]]
    scores_closed = scores + [scores[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_closed,
        theta=families_closed,
        fill="toself",
        name="Affinité sémantique",
        line=dict(color="#6C63FF", width=2),
        fillcolor="rgba(108, 99, 255, 0.25)",
        hovertemplate="%{theta} : %{r:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title="Profil olfactif — Affinité sémantique par famille",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%"),
        ),
        showlegend=False,
        margin=dict(l=50, r=50, t=60, b=40),
    )
    return fig


def plot_top3_breakdown(top_3: list) -> go.Figure:
    """
    Génère un diagramme en barres empilées horizontales montrant
    la contribution de chaque critère au score final du Top 3.
    """
    # Inverser l'ordre pour afficher le #1 tout en haut
    top_3_rev = list(reversed(top_3))
    labels = [f"{p['Name']} ({p['Brand']})" for p in top_3_rev]

    criteria = [
        ("semantic", "Similarité sémantique (SBERT)", "#6C63FF"),
        ("family", "Affinité famille olfactive", "#00B894"),
        ("occasion", "Occasion & Saisons", "#FDCB6E"),
        ("intensity", "Tenue & Sillage", "#E17055"),
    ]

    fig = go.Figure()
    for key, label, color in criteria:
        values = [p["breakdown"].get(key, 0) for p in top_3_rev]
        fig.add_trace(go.Bar(
            name=label,
            y=labels,
            x=values,
            orientation="h",
            marker_color=color,
            text=[f"{v:.0%}" if v >= 0.05 else "" for v in values],
            textposition="inside",
            hovertemplate=f"{label} : %{{x:.1%}}<extra></extra>",
        ))

    fig.update_layout(
        title="Décomposition du score final — Top 3 recommandé",
        barmode="stack",
        xaxis=dict(title="Score global", range=[0, 1], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=60, b=90),
    )
    return fig
