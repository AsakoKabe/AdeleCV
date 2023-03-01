import logging
import uuid

import dash_mantine_components as dmc
from dash import Output, html
from dash.development.base_component import Component
from dash_extensions.enrich import LogConfig


def get_notification_log_writers():
    def _default_kwargs(color, title, message, auto_close=False):
        return {
            'color': color, 'title': title, 'message': message,
            'id': str(uuid.uuid4()),
            'action': "show", 'autoClose': auto_close
        }

    def log_info(message, **kwargs):
        return dmc.Notification(
            **{**_default_kwargs("green", "Info", message, auto_close=True),
               **kwargs}
            )

    def log_debug(message, **kwargs):
        return dmc.Notification(
            **{**_default_kwargs("blue", "Debug", message, auto_close=True),
               **kwargs}
            )

    def log_warning(message, **kwargs):
        return dmc.Notification(
            **{**_default_kwargs("yellow", "Warning", message), **kwargs}
            )

    def log_error(message, **kwargs):
        return dmc.Notification(
            **{**_default_kwargs("red", "Error", message), **kwargs}
            )

    return {
        logging.INFO: log_info, logging.WARNING: log_warning,
        logging.ERROR: log_error, "DEBUG": log_debug,
    }


def setup_notifications_log_config():
    log_id = "notifications_provider"
    log_output = Output(log_id, "children")

    def notification_layout_transform(layout: list[Component]):
        layout.append(html.Div(id=log_id))
        return [dmc.NotificationsProvider(layout, limit=10)]

    return LogConfig(
        log_output, get_notification_log_writers(),
        notification_layout_transform
        )
