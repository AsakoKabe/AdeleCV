import dash_bootstrap_components as dbc

nav = dbc.NavbarSimple(
    children=[
        dbc.NavLink("Dataset", href="/dataset", active="exact"),
        dbc.NavLink("Train", href="/train", active="exact"),
        dbc.NavLink("Table Models", href="/table-models", active="exact"),
        dbc.NavLink("Console", href="/log-console", active="exact"),
    ],
    brand="AdeleCV",
    brand_href="/"
    # color="light",
    # dark=True,
)
