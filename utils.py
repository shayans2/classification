import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


def plot_2d_boundary(model, X, y):
    # Generate grid points within the range of the input data
    u = np.linspace(np.min(X[:, 0])-0.5, np.max(X[:, 0])+0.5, 50)
    v = np.linspace(np.min(X[:, 1])-0.5, np.max(X[:, 1])+0.5, 50)
    
    # Generate meshgrid from the grid points
    U, V = np.meshgrid(u, v)

    # Flatten and concatenate U, V as input features for predictions
    UV = np.c_[U.ravel(), V.ravel()]

    # Make predictions for each point on the grid
    Z = model.predict(UV).reshape(U.shape)

    # Create a new figure and axis for the plot
    _, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot of the input data points colored by the true labels
    sns.scatterplot(data=pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'y': y}),
                    x='X1', y='X2', hue='y', ax=ax)
    
    # Plot the decision boundary contour at level 0.5
    ax.contour(U, V, Z, levels=[0.5], colors='green')

    # Display the plot
    plt.show()



import plotly.graph_objects as go

def plot_3d_boundary(model, X, y):
    # Generate grid points within the range of the input data
    grid_range = np.max(X, axis=0) - np.min(X, axis=0)
    grid_range += np.array([0.5, 0.5, 0.5])  # Add a margin to the grid range
    grid_size = 50
    grid_axes = [np.linspace(np.min(X[:, i]) - grid_range[i], np.max(X[:, i]) + grid_range[i], grid_size) for i in range(X.shape[1])]
    grid = np.meshgrid(*grid_axes)
    xyz = np.column_stack([g.flatten() for g in grid])

    # Compute the predicted probabilities for each point in the grid
    probs = model.predict(xyz).reshape(grid[0].shape)

    # Create the 3D surface plot
    fig = go.Figure(data=go.Volume(
        x=grid[0].flatten(),
        y=grid[1].flatten(),
        z=grid[2].flatten(),
        value=probs.flatten(),
        isomin=0.5,
        isomax=0.5,
        opacity=1,
        surface_count=1,
        colorscale='Jet'
    ))

    # Add the scatter plot of the data points
    fig.add_trace(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=y,
            opacity=0.8,
            colorscale='Portland'
        ),
    ))

    # Set the axis labels and plot title
    fig.update_layout(scene=dict(
        xaxis_title='X Label',
        yaxis_title='Y Label',
        zaxis_title='Z Label',
    ),
        title='Logistic Regression Decision Boundary')

    fig.show()



def get_line_eq(model):
    feature_names = [f'x{i}' for i in range(1, len(model.weights) + 1)]
    coefficients = ' + '.join(f'{coeff} * {feat_name}' for coeff, feat_name in zip(model.weights, feature_names))
    line_eq = f'y = {coefficients}'
    return line_eq

