# main.py
from engine.app import App

if __name__ == '__main__':
    reload_assets = False
    # reload_assets = True

    my_app = App(1920, 1080, 'Roundtable Hold', reload_assets=reload_assets)
    my_app.run()
