import plotly.graph_objects as go


def plot_lines(groups, axes_titles=['x', 'y'], title='Figure'):
    fig = go.Figure()
    for name in groups:
        fig.add_trace(go.Scatter(x=groups[name][0], y=groups[name][1],
                                 mode='lines',
                                 name=name))
    
    fig.update_layout(title=title,
                   xaxis_title=axes_titles[0],
                   yaxis_title=axes_titles[1])
    fig.show()


def plot_bars(groups, axes_titles=['x', 'y'], title='Performance Figure'):
    return
