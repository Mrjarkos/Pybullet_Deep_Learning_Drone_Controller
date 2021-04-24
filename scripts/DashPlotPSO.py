import dash
import logging
import numpy as np
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

logging.getLogger('werkzeug').setLevel(logging.ERROR)

class DashPlotPSO(object):
    def __init__(self, file='./logs/PSOCheckpoints/checkpoint_0.npy'):
        with open(file, 'rb') as f:
            solution =  np.load(f, allow_pickle=True)
            history = solution[()]['history']
        self.n = history['i']
        self.HistVal = history['values']
        self.HistBestVal = history['bestValue']
        self.HistBestPos = history['bestPosition']
        self.HistParPos = history['particlePosition']
        self.HistParVal = history['particleValue']
        self.HistW = history['inertia']
        self.HistEpsilon = history['exploration']
        
        self.x_name = history['parameters']['x_names'] 
        self.x_min = history['parameters']['x_min'] 
        self.x_max = history['parameters']['x_max'] 
        self.n_var = history['parameters']['n_var'] 
        self.exploration = history['parameters']['exploration'] 
        self.N = history['parameters']['N'] 

        app = dash.Dash(__name__)

        app.layout = html.Div([
            html.H2("PSO Dashboard Visualization", style={'text-align': 'center'}),
            html.P("Iteration"),
            dcc.Slider(
                id='epochs',
                min=1,
                max=self.n,
                step=1,
                value=self.n,
                tooltip = { 'always_visible': True }
            ),
            dcc.Graph(id="graph-epochs", style={'display': 'inline-block','width': '98vw', 'height': '75vh'}),
            dcc.Graph(id="graph-x", style={'display': 'inline-block','width': '98vw', 'height': '50vh'}),
            dcc.Graph(id="graph-cost", style={'display': 'inline-block','width': '98vw', 'height': '50vh'}),
        ])

        @app.callback(
            Output("graph-x", "figure"), 
            [Input("epochs", "value")])
        def plot_x(epoch):
            fig = go.Figure()
            for i in range(self.n_var):
                fig.add_trace(
                    go.Scatter(
                        x=[i for i in range(epoch)],
                        y=self.HistBestPos[:epoch, i],
                        marker=dict(
                            size=2,
                        ),
                        mode="lines",
                        name=self.x_name[i]
                    )
                )
            fig.update_layout(
                showlegend=True,
                title_text='PSO Particles Positions',
                xaxis=dict(title="Iterations", range=[0, self.n]),
            )
            fig.update_yaxes(title_text="Best Position per Decision Variable")
            return fig

        @app.callback(
            Output("graph-epochs", "figure"), 
            [Input("epochs", "value")])
        def plot_epochs(epoch):
            fig = go.Figure()
            if epoch>self.n//5:
                init = epoch-self.n//5
            else:
                init=0
            if self.n_var==2:
                for i in range(init,epoch):
                    fig.add_trace(
                        go.Scatter(
                            x=self.HistParPos[i,:,0], y=self.HistParPos[i,:,1], 
                            opacity=0.5*(((i-init)+1)/(epoch-init)),
                            marker=dict(
                                size=8,
                                color=self.HistParVal[epoch,:],
                                colorscale="aggrnyl"
                            ),
                            mode="markers",
                            name='epoch:'+str(i)
                        ),
                    )
                fig.add_trace(
                    go.Scatter(
                        x=self.HistParPos[epoch,:,0], y=self.HistParPos[epoch,:,1], 
                        marker=dict(
                            size=16,
                            color=self.HistParVal[epoch,:],
                            colorbar=dict(
                                title="Cost Value",
                            ),
                            cmin=min(self.HistParVal[self.n,:]),
                            cmax=max(self.HistParVal[0,:]),
                            colorscale="aggrnyl"
                        ),
                        mode="markers",
                        name='epoch:'+str(epoch)
                    )
                )
                fig.update_layout(
                    showlegend=False,
                    title_text='PSO Cost',
                    xaxis=dict(title=self.x_name[0], range=[self.x_min[0], self.x_max[0]]),
                    yaxis=dict(title=self.x_name[1], range=[self.x_min[1], self.x_max[1]]),
                )
            if self.n_var==3:
                for i in range(init,epoch):
                    fig.add_trace(
                        go.Scatter3d(
                            x=self.HistParPos[i,:,0], y=self.HistParPos[i,:,1], z=self.HistParPos[i,:,2],
                            opacity=0.5*(((i-init)+1)/(epoch-init)),
                            marker=dict(
                                size=4,
                                color=self.HistParVal[epoch,:],
                                colorscale="aggrnyl"
                            ),
                            mode="markers",
                            name='epoch:'+str(i)
                        ),
                    )
                fig.add_trace(
                    go.Scatter3d(
                        x=self.HistParPos[epoch,:,0], y=self.HistParPos[epoch,:,1],  z=self.HistParPos[epoch,:,2],
                        opacity=0.8,
                        marker=dict(
                            size=8,
                            color=self.HistParVal[epoch,:],
                            colorbar=dict(
                                title="Cost Value",
                            ),
                            cmin=min(self.HistParVal[self.n,:]),
                            cmax=max(self.HistParVal[0,:]),
                            colorscale="aggrnyl"
                        ),
                        mode="markers",
                        name='epoch:'+str(epoch)
                    )
                )
                fig.update_layout(
                    showlegend=False,
                    title_text='PSO Cost',
                    scene = dict(
                        xaxis=dict(title=self.x_name[0], range=[self.x_min[0], self.x_max[0]]),
                        yaxis=dict(title=self.x_name[1], range=[self.x_min[1], self.x_max[1]]),
                        zaxis=dict(title=self.x_name[2], range=[self.x_min[2], self.x_max[2]])
                    )
                )
            return fig

        @app.callback(
            Output("graph-cost", "figure"), 
            [Input("epochs", "value")])
        def plot_cost(epoch):
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=[i for i in range(epoch)],
                    y=self.HistVal[:epoch],
                    opacity=0.75,
                    marker=dict(
                        size=2,
                        color="red",
                    ),
                    mode="lines",
                    name='Cost Value'
                ),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=[i for i in range(epoch)],
                    y=self.HistBestVal[:epoch],
                    opacity=0.75,
                    marker=dict(
                        color="green",
                    ),
                    mode="lines",
                    name='Best Cost Value'
                ),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=[i for i in range(epoch)],
                    y=self.HistW[:epoch],
                    opacity=0.75,
                    marker=dict(
                        color="blue",
                    ),
                    mode="lines",
                    name='Inertia Value',
                ),
                secondary_y=True
            )
            if self.exploration:
                fig.add_trace(
                    go.Scatter(
                        x=[i for i in range(epoch)],
                        y=self.HistEpsilon[:epoch],
                        opacity=0.75,
                        marker=dict(
                            color="yellow",
                        ),
                        mode="lines",
                        name='Exploration Value',    
                    ),
                    secondary_y=True
                )
            fig.update_layout(
                showlegend=True,
                title_text='PSO Iteration Cost',
                xaxis=dict(title="Iterations", range=[0, self.n]),
            )
            fig.update_yaxes(title_text="Cost per Iteration",
                            tickformat='e.2f',
                            secondary_y=False,
                            range=[min(self.HistVal), max(self.HistVal)])
            fig.update_yaxes(title_text="Inertia/Exploration Value",
                             secondary_y=True,
                             range=[0, max(self.HistW)])
            return fig

        app.run_server(debug=False)

if __name__ == '__main__':
    DashPlotPSO()