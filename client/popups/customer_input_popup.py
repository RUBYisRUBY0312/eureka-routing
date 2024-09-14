import PySimpleGUI as sg

KEY_CUSTOMER_LAT = 'K_CUSTOMER_LAT'
KEY_CUSTOMER_LONG = 'K_CUSTOMER_LONG'
KEY_CUSTOMER_DEMAND = 'K_CUSTOMER_DEMAND'
KEY_CUSTOMER_ERROR = 'K_CUSTOMER_ERROR'
KEY_CUSTOMER_CONFIRM = 'K_CUSTOMER_CONFIRM'

def customer_input_popup(id = None, lat = None, long = None, demand = None):
    layout = [
        [
            sg.Column([
                [sg.Text('Latitude', size=(20, 1)), sg.Text('Longitude', size=(20, 1))],
                [sg.Input(lat, size=(20, 1), k=KEY_CUSTOMER_LAT), sg.Input(long, size=(20, 1), k=KEY_CUSTOMER_LONG)],
                [sg.Text('Demand', size=(20, 1))],
                [sg.Input(demand, size=(20, 1), k=KEY_CUSTOMER_DEMAND)],
                [sg.Text('Error: invalid values', k=KEY_CUSTOMER_ERROR, visible=False, colors='red')]
            ]),
        ],
        [
            sg.Push(),
            sg.Button('Save' if id is not None else 'Add customer', k=KEY_CUSTOMER_CONFIRM, enable_events=True),
            sg.Push()
        ]
    ]

    title = f'Edit customer #{id}' if id is not None else 'Add customer'
    popup = sg.Window(title, layout=layout, modal=True)

    while True:
        popup_events, popup_values = popup.read()
        if popup_events == sg.WIN_CLOSED:
            return None

        if popup_events == KEY_CUSTOMER_CONFIRM:
            try:
                lat = float(popup_values[KEY_CUSTOMER_LAT])
                long = float(popup_values[KEY_CUSTOMER_LONG])
                demand = int(popup_values[KEY_CUSTOMER_DEMAND])
                popup[KEY_CUSTOMER_ERROR].update(visible=False)
                popup.close()
                return [lat, long, demand]
            except ValueError:
                popup[KEY_CUSTOMER_ERROR].update(visible=True)