# =============================================================================
# Drawdown Metrics
# =============================================================================

def calculate_drawdown_metrics(returns):
    """
    Compute Maximum Drawdown, Average Drawdown,
    and Maximum Underwater Duration.
    """

    wealth = (
        1.0 + returns
    ).cumprod()

    running_max = (
        wealth.cummax()
    )

    drawdown = (
        wealth / running_max
    ) - 1.0

    # -------------------------------------------------------------------------
    # Maximum Drawdown
    # -------------------------------------------------------------------------

    max_dd = drawdown.min()

    # -------------------------------------------------------------------------
    # Average Drawdown
    # -------------------------------------------------------------------------

    negative_drawdowns = (
        drawdown[
            drawdown < 0
        ]
    )

    avg_dd = (
        negative_drawdowns.mean()
        if len(negative_drawdowns) > 0
        else 0.0
    )

    # -------------------------------------------------------------------------
    # Maximum Underwater Duration
    # -------------------------------------------------------------------------

    max_duration = 0
    current_duration = 0

    for dd in drawdown:

        if dd < 0:

            current_duration += 1

        else:

            max_duration = max(
                max_duration,
                current_duration,
            )

            current_duration = 0

    max_duration = max(
        max_duration,
        current_duration,
    )

    return (
        max_dd,
        avg_dd,
        max_duration,
    )