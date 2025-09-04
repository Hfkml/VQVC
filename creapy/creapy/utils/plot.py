from __future__ import annotations

from .helpers import get_time_vector
from .config import get_config
# from ..feature_extraction import get_feature_list

import numpy as np
import pandas as pd

def plot(X_test: pd.DataFrame, 
         y_pred: np.ndarray,
         sr: int,
         title: str | None = None,
         words: list | None = None):
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    _config = get_config()['USER']
    #t0 = _config['audio_start']
    t0 = float(words[0]['start'])-.02 if words else 0.0
    #features = X_test.columns.to_list()
    df = pd.concat(
        (pd.Series(y_pred, name='creak_probability'), X_test), axis=1
    )
    df['creak_threshold'] = _config['creak_threshold']
   
    df = df.drop(columns=['zcr', 'ste', 'hnr', 'jitter', 'shimmer', 'f0mean'])
    features = ['h1h2']
    #drop rows before t0, note that t0 is in seconds and the columns are in windowed frames
    first_index = int(t0//_config['hop_size'])-1
    last_index = int(words[-1]['end']//_config['hop_size'])-3
    df = df.iloc[first_index:last_index]
    df_norm = df.copy()
    df_norm[features] = df[features].apply(lambda x: (x/x.abs().max()+1)/2, axis=0)
    print(t0)
    #dt = get_time_vector(y_pred, sr, t0)
    dt = get_time_vector(y_pred[first_index:last_index], sr, t0)
    
    fig = px.line(df_norm, 
                  x=dt, 
                  y=df_norm.columns,
                  title=title)

    # Add text annotations
    if words:
        for word in words:
            start_time = word['start']
            end_time = word['end']
            text = word['word']
            
            fig.add_shape(
                dict(
                    type="rect",
                    x0=start_time,
                    y0=-0.1,  # Adjust this value based on the y-axis range
                    x1=end_time,
                    y1=1.1,   # Adjust this value based on the y-axis range
                    fillcolor="LightSkyBlue",
                    opacity=0.5,
                    layer="below",
                    line=dict(width=0)
                )
            )
            
            fig.add_annotation(
                x=(start_time + end_time) / 2,
                y=1.05,  # Adjust this value based on the y-axis range
                text=text,
                showarrow=False,
                font=dict(size=12)
            )

    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        template="plotly_white",
    )
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            title='Time [s]'
        )
    )
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"y": [df_norm[column] for column in df_norm.columns]}],
                        label="Normalized",
                        method="update"
                    ),
                ]),
                pad={"r": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    
    if title:
        fig.update_layout(
            title={
                'text': title,
                'y': 0.99,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

    fig.show()
    return fig
