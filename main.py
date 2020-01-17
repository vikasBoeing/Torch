'''this is the main file'''
from src.torch import *
# import src.cnn as cnn
# from src.drl.mc import run
from src.drl.TD import run
if __name__ == '__main__':
    print("Hello from main")
    run()