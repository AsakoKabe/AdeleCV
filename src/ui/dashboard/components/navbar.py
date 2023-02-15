import dash_bootstrap_components as dbc


nav = dbc.NavbarSimple(
    children=[
        dbc.NavLink("Dataset", href="/dataset", active="exact"),
        dbc.NavLink("Train", href="/train", active="exact"),
        dbc.NavLink("Table Models", href="/table-models", active="exact"),
    ],
    brand="AutoDL-CV",
    brand_href="/"
    # color="light",
    # dark=True,
)
