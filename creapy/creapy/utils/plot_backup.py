from __future__ import annotations

from .helpers import get_time_vector
from .config import get_config
# from ..feature_extraction import get_feature_list

import numpy as np
import pandas as pd

def plot(X_test: pd.DataFrame, 
         y_pred: np.ndarray,
         sr: int,
         title: str | None = None):
    import plotly.express as px
    _config = get_config()['USER']
    t0 = _config['audio_start']
    features = X_test.columns.to_list()
    df = pd.concat(
        (pd.Series(y_pred, name='creak_probability'), X_test), axis=1
    )
    df['creak_threshold'] = _config['creak_threshold']
    print(df.columns)
   
    #Index(['creak_probability', 'zcr', 'ste', 'hnr', 'jitter', 'h1h2', 'shimmer',
    #   'f0mean', 'creak_threshold'],
    #  dtype='object')
     #drop df['zcr'], df['ste'], df['hnr'], jitter, h1h2, shimmer
    df = df.drop(columns=['zcr', 'ste', 'hnr', 'jitter', 'h1h2', 'shimmer'])
    #drop from features
    features = ['f0mean']



    df_norm = df.copy()
    df_norm[features] = df[features].apply(lambda x: x/x.abs().max(), axis=0)
    dt = get_time_vector(y_pred, sr, t0)
    
    
    fig = px.line(df_norm, 
                  x=dt, 
                  y=df_norm.columns,
                  title=title)

    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        template="plotly_white",
    )
    fig.update_layout(
        xaxis = dict(
            rangeslider = dict(
                visible = True
            ),
            title='Time [s]'
        )
    )
    fig.update_layout(
        updatemenus=[
            dict(
                type = "buttons",
                direction = "left",
                buttons=list([
                    dict(
                        args=[{"y": [df_norm[column] for column in df_norm.columns]}],
                        label="Normalized",
                        method="update"
                    ),
#                    dict(
#                        args=[{"y": [df[column] for column in df.columns]}],
#                        label="Original",
#                        method="update"
#                    ),
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
            'y':0.99,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
        )

    fig.show()
    return fig
    
    