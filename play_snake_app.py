

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
import sys
from typing import List
from snake import *
import numpy as np
from nn_viz import NeuralNetworkViz
from neural_network import FeedForwardNetwork, sigmoid, linear, relu
from settings import settings
from genetic_algorithm.population import Population
from genetic_algorithm.selection import elitism_selection, roulette_wheel_selection, tournament_selection
from genetic_algorithm.mutation import gaussian_mutation, random_uniform_mutation
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.crossover import uniform_binary_crossover, single_point_binary_crossover
from math import sqrt
from decimal import Decimal
import random
import csv
import time

current_milli_time = lambda: int(round(time.time() * 1000))


SQUARE_SIZE = (35, 35)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings, show=True, fps=200):
        super().__init__()
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor(240, 240, 240))
        self.setPalette(palette)
        self.settings = settings
        self._SBX_eta = self.settings['SBX_eta']
        
        self.board_size = settings['board_size']
        self.border = (0, 10, 0, 10)  # Left, Top, Right, Bottom
        self.snake_widget_width = SQUARE_SIZE[0] * self.board_size[0]
        self.snake_widget_height = SQUARE_SIZE[1] * self.board_size[1]

        # Allows padding of the other elements even if we need to restrict the size of the play area
        self._snake_widget_width = max(self.snake_widget_width, 620)
        self._snake_widget_height = max(self.snake_widget_height, 600)

        self.top = 150
        self.left = 150
        self.width = self._snake_widget_width + 700 + self.border[0] + self.border[2]
        self.height = self._snake_widget_height + self.border[1] + self.border[3] + 200
        
        self.best_fitness = 0
        self.best_score = 0
        self.best_snake = None

        self._current_individual = 0

        self.snake = load_snake('saves/', 'snake-36-620')
        print(self.snake.is_alive)
        self.current_generation = 0

        self.init_window()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(100)

        self.timeBeforeGen = current_milli_time()

        if show:
            self.show()

    def init_window(self):
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle('Snake AI')
        self.setGeometry(self.top, self.left, self.width, self.height)

        # Create the Neural Network window
        self.nn_viz_window = NeuralNetworkViz(self.centralWidget, self.snake)
        self.nn_viz_window.setGeometry(QtCore.QRect(0, 0, 600, self._snake_widget_height + self.border[1] + self.border[3] + 200))
        self.nn_viz_window.setObjectName('nn_viz_window')

        # Create SnakeWidget window
        self.snake_widget_window = SnakeWidget(self.centralWidget, self.board_size, self.snake)
        self.snake_widget_window.setGeometry(QtCore.QRect(600 + self.border[0], self.border[1], self.snake_widget_width, self.snake_widget_height))
        self.snake_widget_window.setObjectName('snake_widget_window')


    def update(self) -> None:
        print(self.snake)
        self.snake_widget_window.update()
        self.nn_viz_window.update()
        # Current individual is alive
        if self.snake.is_alive:
            print('moving snake')
            self.snake.move()
            if self.snake.score > self.best_score:
                self.best_score = self.snake.score
        else:
            print('snake  is dead')
        # Current individual is dead         

class SnakeWidget(QtWidgets.QWidget):
    def __init__(self, parent, board_size=(50, 50), snake=None):
        super().__init__(parent)
        self.board_size = board_size
        # self.setFixedSize(SQUARE_SIZE[0] * self.board_size[0], SQUARE_SIZE[1] * self.board_size[1])
        # self.new_game()
        if snake:
            self.snake = snake
        self.setFocus()

        self.draw_vision = True
        self.show()

    def new_game(self) -> None:
        self.snake = Snake(self.board_size)
    
    def update(self):
        if self.snake.is_alive:
            self.snake.update()
            self.repaint()
        else:
            # dead
            pass

    def draw_border(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QtGui.QPen(Qt.black))
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        painter.setPen(QtCore.Qt.black)
        painter.drawLine(0, 0, width, 0)
        painter.drawLine(width, 0, width, height)
        painter.drawLine(0, height, width, height)
        painter.drawLine(0, 0, 0, height)

    def draw_snake(self, painter: QtGui.QPainter) -> None:
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor(0, 0, 0))
        # painter.setPen(QtGui.QPen(Qt.black))
        painter.setPen(pen)
        brush = QtGui.QBrush()
        brush.setColor(Qt.red)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(198, 5, 20)))
        # painter.setBrush(brush)

        def _draw_line_to_apple(painter: QtGui.QPainter, start_x: int, start_y: int, drawable_vision: DrawableVision) -> Tuple[int, int]:
            painter.setPen(QtGui.QPen(Qt.green))
            end_x = drawable_vision.apple_location.x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
            end_y = drawable_vision.apple_location.y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2
            painter.drawLine(start_x, start_y, end_x, end_y)
            return end_x, end_y

        def _draw_line_to_self(painter: QtGui.QPainter, start_x: int, start_y: int, drawable_vision: DrawableVision) -> Tuple[int, int]:
            painter.setPen(QtGui.QPen(Qt.red))
            end_x = drawable_vision.self_location.x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
            end_y = drawable_vision.self_location.y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2
            painter.drawLine(start_x, start_y, end_x, end_y)
            return end_x, end_y

        for point in self.snake.snake_array:
            painter.drawRect(point.x * SQUARE_SIZE[0],  # Upper left x-coord
                             point.y * SQUARE_SIZE[1],  # Upper left y-coord
                             SQUARE_SIZE[0],            # Width
                             SQUARE_SIZE[1])            # Height

        if self.draw_vision:
            start = self.snake.snake_array[0]

            if self.snake._drawable_vision[0]:
                for drawable_vision in self.snake._drawable_vision:
                    start_x = start.x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
                    start_y = start.y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2
                    if drawable_vision.apple_location and drawable_vision.self_location:
                        apple_dist = self._calc_distance(start.x, drawable_vision.apple_location.x, start.y, drawable_vision.apple_location.y)
                        self_dist = self._calc_distance(start.x, drawable_vision.self_location.x, start.y, drawable_vision.self_location.y)
                        if apple_dist <= self_dist:
                            start_x, start_y = _draw_line_to_apple(painter, start_x, start_y, drawable_vision)
                            start_x, start_y = _draw_line_to_self(painter, start_x, start_y, drawable_vision)
                        else:
                            start_x, start_y = _draw_line_to_self(painter, start_x, start_y, drawable_vision)
                            start_x, start_y = _draw_line_to_apple(painter, start_x, start_y, drawable_vision)

                    elif drawable_vision.apple_location:
                        start_x, start_y = _draw_line_to_apple(painter, start_x, start_y, drawable_vision)

                    elif drawable_vision.self_location:
                        start_x, start_y = _draw_line_to_self(painter, start_x, start_y, drawable_vision)
                        
                    if drawable_vision.wall_location:
                        painter.setPen(QtGui.QPen(Qt.black))
                        end_x = drawable_vision.wall_location.x * SQUARE_SIZE[0] + SQUARE_SIZE[0]/2
                        end_y = drawable_vision.wall_location.y * SQUARE_SIZE[1] + SQUARE_SIZE[1]/2 
                        painter.drawLine(start_x, start_y, end_x, end_y)


    def draw_apple(self, painter: QtGui.QPainter) -> None:
        apple_location = self.snake.apple_location
        if apple_location:
            painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
            painter.setPen(QtGui.QPen(Qt.black))
            painter.setBrush(QtGui.QBrush(Qt.green))

            painter.drawRect(apple_location.x * SQUARE_SIZE[0],
                             apple_location.y * SQUARE_SIZE[1],
                             SQUARE_SIZE[0],
                             SQUARE_SIZE[1])

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        painter.begin(self)

        self.draw_border(painter)
        self.draw_apple(painter)
        self.draw_snake(painter)
        
        painter.end()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key_press = event.key()
        # if key_press == Qt.Key_Up:
        #     self.snake.direction = 'u'
        # elif key_press == Qt.Key_Down:
        #     self.snake.direction = 'd'
        # elif key_press == Qt.Key_Right:
        #     self.snake.direction = 'r'
        # elif key_press == Qt.Key_Left:
        #     self.snake.direction = 'l'

    def _calc_distance(self, x1, x2, y1, y2) -> float:
        diff_x = float(abs(x2-x1))
        diff_y = float(abs(y2-y1))
        dist = ((diff_x * diff_x) + (diff_y * diff_y)) ** 0.5
        return dist

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(settings)
    sys.exit(app.exec_())
