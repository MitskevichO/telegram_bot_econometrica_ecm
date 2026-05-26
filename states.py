
from aiogram.fsm.state import State, StatesGroup


class Flow(StatesGroup):
    waiting_file = State()
    choosing_roles = State()
    entering_meta_y = State()
    entering_meta_x = State()
    ready = State()
    waiting_x_for_ci = State()
    selecting_ci = State()
    selecting_horizon = State()
    scenario_input = State()
    selecting_lags_dy = State()
    waiting_chow_year = State()
    waiting_x_for_ecm_forecast = State()
